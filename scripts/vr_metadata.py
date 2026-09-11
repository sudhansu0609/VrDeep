"""
Spherical Video V2 metadata for MP4 - what makes a headset treat a file as VR.

WHY THIS EXISTS
---------------
The pipeline produced a correct 8192x4096 side-by-side VR180 file and the Quest
played it as a flat 2D video on a curved screen. Nothing was wrong with the
picture; the file simply never said what it was.

An MP4 carries that claim in two boxes inside the video sample entry:

    st3d   how the two eyes are packed (left-right / top-bottom / mono)
    sv3d   how the pixels map onto a sphere (equirectangular, and how much
           of the sphere the frame covers)

Neither was present - not in our output, and not in the SLR source either.
Checked with ffprobe: no side data, no st3d, no sv3d, no uuid. The source plays
correctly on the headset only because it is *named* `..._LR.mp4`, and DeoVR,
Skybox and Pigasus parse that convention. Our output was named
`vr_deepfake_20260905_081929.mp4`, which tells those players nothing, and tells
a metadata-driven player (Quest's own media player, YouTube VR) nothing either.

`engine.apply_stereo_metadata` could not have fixed this. It writes Matroska's
`stereo_mode` tag, which ffmpeg silently drops for MP4 - and even in an MKV that
tag only describes eye packing, never the projection, so the result would still
have been a flat stereo pair rather than a sphere.

ffmpeg can read and write these boxes (avformat has st3d/sv3d/equi/prhd/svhd),
but only by *propagating* spherical side data from an input that already has it.
There is no CLI option to originate it, so we write the boxes ourselves.

WHAT IT WRITES
--------------
Google's Spherical Video V2 layout, appended to the video sample entry:

    st3d                          stereo packing
    sv3d
      svhd                        who wrote this
      proj
        prhd                      pose - always zero here
        equi                      equirectangular + how much sphere is covered

For VR180 the frame holds the middle half of a full 360 equirect, so `equi`
crops 25% off each side (`projection_bounds_left/right = 0.25`). For a full 360
frame every bound is zero.

SAFETY
------
Only `moov` is read into memory - a few MB of a file that may be a gigabyte -
and the rest is streamed through. Inserting bytes inside `moov` moves anything
that follows it, so every `stco`/`co64` chunk offset at or past the insertion
point is corrected. When `moov` sits after `mdat`, which is how the pipeline's
own files come out, no offset moves at all.
"""

import io
import os
import shutil
import struct

# st3d stereo_mode values, from the Spherical Video V2 spec.
STEREO_MONO = 0
STEREO_TOP_BOTTOM = 1
STEREO_LEFT_RIGHT = 2

LAYOUT_TO_STEREO = {
    "sbs": STEREO_LEFT_RIGHT,
    "lr": STEREO_LEFT_RIGHT,
    "tb": STEREO_TOP_BOTTOM,
    "ou": STEREO_TOP_BOTTOM,
    "mono": STEREO_MONO,
}

# projection_bounds_* are 0.32 fixed point: the fraction cropped from each edge.
_HALF_SPHERE_CROP = 0x40000000  # 0.25 * 2**32

VIDEO_HANDLER = b"vide"


class MetadataError(Exception):
    """The file is not shaped the way this writer knows how to patch."""


# ---------------------------------------------------------------------------
# Box reading
# ---------------------------------------------------------------------------

def _iter_boxes(buf, start, end):
    """Yield (offset, size, type, header_len) for each box in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", buf[pos:pos + 4])[0]
        box_type = buf[pos + 4:pos + 8]
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield (pos, size, box_type, header)
        pos += size


def _find_box(buf, start, end, box_type):
    for pos, size, typ, header in _iter_boxes(buf, start, end):
        if typ == box_type:
            return (pos, size, header)
    return None


def _require(found, what, where):
    if found is None:
        raise MetadataError(f"no {what} box inside {where}")
    return found


def _top_level_box(path, wanted):
    """
    (offset, size) of a top-level box, read by seeking rather than by loading.

    The point is not to pull a gigabyte of mdat into memory just to reach the
    few MB of moov that sits behind it.
    """
    total = os.path.getsize(path)
    with io.open(path, "rb") as handle:
        pos = 0
        while pos + 8 <= total:
            handle.seek(pos)
            header_bytes = handle.read(16)
            if len(header_bytes) < 8:
                return None
            size = struct.unpack(">I", header_bytes[0:4])[0]
            box_type = header_bytes[4:8]
            if size == 1:
                if len(header_bytes) < 16:
                    return None
                size = struct.unpack(">Q", header_bytes[8:16])[0]
            elif size == 0:
                size = total - pos
            if size < 8:
                return None
            if box_type == wanted:
                return (pos, size)
            pos += size
    return None


# ---------------------------------------------------------------------------
# Box writing
# ---------------------------------------------------------------------------

def _box(box_type, payload):
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _full_box(box_type, payload, version=0, flags=0):
    return _box(box_type, struct.pack(">BBBB", version,
                                      (flags >> 16) & 0xFF,
                                      (flags >> 8) & 0xFF,
                                      flags & 0xFF) + payload)


def build_st3d(stereo_mode):
    return _full_box(b"st3d", struct.pack(">B", stereo_mode))


def build_sv3d(half_sphere, source="VrDeep"):
    """
    sv3d container: who wrote it, plus an equirectangular projection.

    `half_sphere` is the VR180 case - the frame covers the middle 180 degrees
    of a 360 equirect, so a quarter is cropped from the left and right edges.
    """
    svhd = _full_box(b"svhd", source.encode("utf-8") + b"\x00")

    # Pose is yaw/pitch/roll as 16.16 fixed point. The camera is the origin.
    prhd = _full_box(b"prhd", struct.pack(">iii", 0, 0, 0))

    crop = _HALF_SPHERE_CROP if half_sphere else 0
    equi = _full_box(b"equi", struct.pack(">IIII", 0, 0, crop, crop))

    return _box(b"sv3d", svhd + _box(b"proj", prhd + equi))


# ---------------------------------------------------------------------------
# Locating the video sample entry
# ---------------------------------------------------------------------------

def _is_video_trak(moov, trak_pos, trak_size, trak_header):
    mdia = _find_box(moov, trak_pos + trak_header, trak_pos + trak_size, b"mdia")
    if mdia is None:
        return None
    mdia_pos, mdia_size, mdia_header = mdia

    hdlr = _find_box(moov, mdia_pos + mdia_header, mdia_pos + mdia_size, b"hdlr")
    if hdlr is None:
        return None
    hdlr_pos, hdlr_size, hdlr_header = hdlr
    # hdlr: version/flags(4), pre_defined(4), handler_type(4)
    handler = moov[hdlr_pos + hdlr_header + 8:hdlr_pos + hdlr_header + 12]
    if handler != VIDEO_HANDLER:
        return None
    return (mdia_pos, mdia_size, mdia_header)


def _sample_entry_chain(moov, moov_header):
    """
    Byte ranges of every box from moov down to the video sample entry.

    Every one of them holds a size field that has to grow by the same amount,
    which is the only reason the whole chain is returned rather than just the
    sample entry.
    """
    moov_end = len(moov)

    for trak_pos, trak_size, typ, trak_header in _iter_boxes(moov, moov_header, moov_end):
        if typ != b"trak":
            continue
        found = _is_video_trak(moov, trak_pos, trak_size, trak_header)
        if found is None:
            continue
        mdia_pos, mdia_size, mdia_header = found

        minf_pos, minf_size, minf_header = _require(
            _find_box(moov, mdia_pos + mdia_header, mdia_pos + mdia_size, b"minf"),
            "minf", "mdia")
        stbl_pos, stbl_size, stbl_header = _require(
            _find_box(moov, minf_pos + minf_header, minf_pos + minf_size, b"stbl"),
            "stbl", "minf")
        stsd_pos, stsd_size, stsd_header = _require(
            _find_box(moov, stbl_pos + stbl_header, stbl_pos + stbl_size, b"stsd"),
            "stsd", "stbl")

        # stsd is a FullBox: version/flags(4) then entry_count(4).
        entries_start = stsd_pos + stsd_header + 8
        entry_count = struct.unpack(
            ">I", moov[stsd_pos + stsd_header + 4:stsd_pos + stsd_header + 8])[0]
        if entry_count < 1:
            raise MetadataError("video stsd has no sample entries")

        entry = next(_iter_boxes(moov, entries_start, stsd_pos + stsd_size), None)
        if entry is None:
            raise MetadataError("video stsd entry could not be read")
        entry_pos, entry_size, entry_type, _ = entry

        return {
            "sample_entry_format": entry_type.decode("ascii", "replace"),
            "chain": [
                (0, len(moov)),          # moov itself, patched via its own header
                (trak_pos, trak_size),
                (mdia_pos, mdia_size),
                (minf_pos, minf_size),
                (stbl_pos, stbl_size),
                (stsd_pos, stsd_size),
                (entry_pos, entry_size),
            ],
            "insert_at": entry_pos + entry_size,
            "existing": _existing_boxes(moov, entry_pos, entry_size),
        }

    raise MetadataError("no video track found in moov")


def _existing_boxes(moov, entry_pos, entry_size):
    """Which spherical boxes the sample entry already carries."""
    present = set()
    # VisualSampleEntry: 8 byte header + 78 bytes of fixed fields, then children.
    for _pos, _size, typ, _header in _iter_boxes(
            moov, entry_pos + 8 + 78, entry_pos + entry_size):
        if typ in (b"st3d", b"sv3d"):
            present.add(typ.decode("ascii"))
    return present


# ---------------------------------------------------------------------------
# Chunk offsets
# ---------------------------------------------------------------------------

def _shift_chunk_offsets(moov, moov_start, insert_at_abs, delta):
    """
    Correct every chunk offset that the insertion pushed forward.

    stco/co64 hold absolute file offsets into mdat. Inserting bytes at
    `insert_at_abs` moves everything at or past it, so those entries - and only
    those - gain `delta`. When moov follows mdat, which is the usual layout
    here, nothing qualifies and this is a no-op.
    """
    if delta == 0:
        return moov, 0

    buf = bytearray(moov)
    shifted = 0

    def walk(start, end):
        nonlocal shifted
        for pos, size, typ, header in _iter_boxes(buf, start, end):
            if typ == b"stco":
                count = struct.unpack(">I", buf[pos + header + 4:pos + header + 8])[0]
                base = pos + header + 8
                for i in range(count):
                    at = base + i * 4
                    value = struct.unpack(">I", buf[at:at + 4])[0]
                    if value >= insert_at_abs:
                        struct.pack_into(">I", buf, at, value + delta)
                        shifted += 1
            elif typ == b"co64":
                count = struct.unpack(">I", buf[pos + header + 4:pos + header + 8])[0]
                base = pos + header + 8
                for i in range(count):
                    at = base + i * 8
                    value = struct.unpack(">Q", buf[at:at + 8])[0]
                    if value >= insert_at_abs:
                        struct.pack_into(">Q", buf, at, value + delta)
                        shifted += 1
            elif typ in (b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"):
                walk(pos + header, pos + size)

    walk(8, len(buf))
    return bytes(buf), shifted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe(path):
    """
    What VR metadata `path` already carries.

    Returns a dict with 'st3d'/'sv3d' booleans and the sample entry format, or
    None when the file cannot be read as an MP4.
    """
    try:
        top = _top_level_box(path, b"moov")
        if top is None:
            return None
        moov_start, moov_size = top
        with io.open(path, "rb") as handle:
            handle.seek(moov_start)
            moov = handle.read(moov_size)
        header = 16 if struct.unpack(">I", moov[0:4])[0] == 1 else 8
        info = _sample_entry_chain(moov, header)
    except (OSError, MetadataError, struct.error):
        return None
    return {
        "format": info["sample_entry_format"],
        "st3d": "st3d" in info["existing"],
        "sv3d": "sv3d" in info["existing"],
    }


def half_sphere_for(width, height, layout):
    """
    True when one eye is a 180-degree frame rather than a full 360 one.

    A full equirect sphere is twice as wide as it is tall. So a single eye that
    comes out square covers 180 degrees horizontally, and one that comes out
    2:1 covers the whole 360. This is what tells `equi` how much of the sphere
    the pixels actually are, and getting it wrong stretches the world.
    """
    if layout in ("tb", "ou"):
        eye_w, eye_h = width, max(1, height // 2)
    elif layout in ("sbs", "lr"):
        eye_w, eye_h = max(1, width // 2), height
    else:
        eye_w, eye_h = width, height
    return (eye_w / float(eye_h)) < 1.5


def inject(path, layout="sbs", half_sphere=True, log_cb=None):
    """
    Write st3d + sv3d into `path`, in place. Returns True if the file changed.

    Raises MetadataError if the file is not an MP4 this can patch; the caller
    is expected to treat that as "leave it alone and say so", not as a failure
    of the run.
    """
    stereo_mode = LAYOUT_TO_STEREO.get(str(layout).lower(), STEREO_LEFT_RIGHT)

    top = _top_level_box(path, b"moov")
    if top is None:
        raise MetadataError("no moov box - not an MP4/MOV file")
    moov_start, moov_size = top

    with io.open(path, "rb") as handle:
        handle.seek(moov_start)
        moov = handle.read(moov_size)
    if len(moov) != moov_size:
        raise MetadataError("moov box is truncated")

    moov_header = 16 if struct.unpack(">I", moov[0:4])[0] == 1 else 8
    info = _sample_entry_chain(moov, moov_header)

    if info["existing"] >= {"st3d", "sv3d"}:
        if log_cb:
            log_cb("VR metadata is already present; leaving the file alone.")
        return False

    payload = build_st3d(stereo_mode) + build_sv3d(half_sphere)
    delta = len(payload)
    insert_at = info["insert_at"]

    patched = bytearray(moov)
    patched[insert_at:insert_at] = payload

    # Every ancestor of the sample entry got longer by the same amount. A
    # 64-bit-size box would need its length written 8 bytes further in; none of
    # these ever are, and silently mis-writing one would corrupt the file.
    for box_start, box_size in info["chain"]:
        if struct.unpack(">I", patched[box_start:box_start + 4])[0] == 1:
            raise MetadataError(
                "this file uses 64-bit box sizes in the moov chain, which this "
                "writer does not handle"
            )
        struct.pack_into(">I", patched, box_start, box_size + delta)

    final, shifted = _shift_chunk_offsets(
        bytes(patched), moov_start, moov_start + insert_at, delta)

    tmp = path + ".vrmeta.tmp"
    total = os.path.getsize(path)
    try:
        with io.open(path, "rb") as src, io.open(tmp, "wb") as dst:
            _copy_range(src, dst, 0, moov_start)
            dst.write(final)
            src.seek(moov_start + moov_size)
            _copy_range(src, dst, moov_start + moov_size, total)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    if log_cb:
        packing = {STEREO_LEFT_RIGHT: "side-by-side",
                   STEREO_TOP_BOTTOM: "over-under",
                   STEREO_MONO: "monoscopic"}[stereo_mode]
        log_cb(
            f"Tagged as VR: {packing}, equirectangular "
            f"{'180' if half_sphere else '360'} degrees."
            + (f" Adjusted {shifted} chunk offsets." if shifted else "")
        )
    return True


def _copy_range(src, dst, start, end):
    src.seek(start)
    remaining = end - start
    chunk = 8 * 1024 * 1024
    while remaining > 0:
        data = src.read(min(chunk, remaining))
        if not data:
            break
        dst.write(data)
        remaining -= len(data)


# ---------------------------------------------------------------------------
# Filename conventions
# ---------------------------------------------------------------------------

# DeoVR, Skybox and Pigasus decide layout from the filename before they look at
# anything else, and SLR names its downloads this way. Metadata alone leaves
# those players guessing, so the suffix is worth carrying even now that the
# boxes are written.
def suffix_for(layout, half_sphere=True):
    packing = "TB" if str(layout).lower() in ("tb", "ou") else "LR"
    return f"_{packing}_{'180' if half_sphere else '360'}"


def name_with_convention(path, layout, half_sphere=True):
    """`out.mp4` -> `out_LR_180.mp4`, without doubling a suffix already there."""
    root, ext = os.path.splitext(path)
    suffix = suffix_for(layout, half_sphere)
    if root.upper().endswith(suffix.upper()):
        return path
    return root + suffix + ext
