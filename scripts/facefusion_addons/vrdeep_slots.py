"""
VrDeep: character slots - which face in the frame gets which source face.

Installed into facefusion/facefusion/ by scripts/patch_facefusion.py, which
also hooks it into face_selector.select_faces and the face swapper's
process_frame. Inactive unless engine.run_facefusion points VRDEEP_FACE_SLOTS
at a slot file, so a plain FaceFusion run behaves exactly as upstream.

Why this exists: upstream 'one' mode takes the largest face *in each frame*,
independently. With two people on screen, whoever is closer to the camera gets
the swap, so across a video the face hops from one character to the other and
both end up swapped. And upstream has no way to put a second source face on a
second person at all - every selected face gets the same averaged source.

Here every slot (Face 1, Face 2) is tied to one character by identity:

  * A slot with reference crops (picked in the app's character picker) matches
    faces by ArcFace embedding against those crops, so it follows that person
    however they move, whoever is larger, across cuts, eyes and chunks.
  * A slot with no references locks onto a character on the first frame it can
    - the largest face of the slot's gender - and from then on follows that
    person's embedding the same way. The lock is written to a file the engine
    names, so the other eye and later chunks reuse it instead of re-locking.

Faces are assigned one-to-one, closest first: a face can only take one source,
and one slot can only take one face per frame, so two characters can never
both get Face 1 and a lookalike bystander loses to the real match.
"""

import json
import os
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy

SLOTS_ENV = 'VRDEEP_FACE_SLOTS'

_lock = threading.Lock()
# slot index -> embeddings it was auto-locked onto (only slots with no references)
_locked : Dict[int, List[numpy.ndarray]] = {}
_lock_file_loaded = False


def emit(message : str) -> None:
	# The engine picks these lines out of FaceFusion's output and puts them in
	# the run log. The leading newline matters: tqdm redraws its progress bar
	# with a bare carriage return, and a report printed straight after one
	# would be glued onto the end of the progress line and never recognised.
	print('\nVRDEEP_SLOT ' + message, flush = True)


@lru_cache(maxsize = None)
def get_spec() -> Optional[dict]:
	path = os.environ.get(SLOTS_ENV)

	if not path:
		return None
	with open(path, encoding = 'utf-8') as handle:
		return json.load(handle)


def is_active() -> bool:
	return get_spec() is not None


def get_slots() -> List[dict]:
	return get_spec().get('slots') or []


def normalize(embedding : numpy.ndarray) -> numpy.ndarray:
	embedding = numpy.asarray(embedding, dtype = numpy.float64).ravel()
	norm = numpy.linalg.norm(embedding)
	return embedding / norm if norm else embedding


@lru_cache(maxsize = None)
def get_reference_embeddings(slot_index : int) -> Tuple[numpy.ndarray, ...]:
	"""
	The embeddings a slot's reference crops stand for.

	The picker writes an .npy of the embedding next to each crop it saves, so
	the usual path is a file read. A crop without one (hand-made, or copied
	without its .npy) is run through the detector instead.
	"""
	from facefusion.face_creator import get_static_faces
	from facefusion.face_selector import sort_faces_by_order
	from facefusion.vision import read_static_image

	embeddings = []

	for path in get_slots()[slot_index].get('references') or []:
		npy_path = os.path.splitext(path)[0] + '.npy'

		if os.path.isfile(npy_path):
			embeddings.append(normalize(numpy.load(npy_path)))
			continue
		vision_frame = read_static_image(path)
		faces = sort_faces_by_order(get_static_faces([ vision_frame ]), 'large-small') if vision_frame is not None else []

		if faces:
			embeddings.append(normalize(faces[0].embedding_norm))
		else:
			emit(f'WARNING Face {slot_index + 1}: no face found in reference {os.path.basename(path)}; ignoring it')
	return tuple(embeddings)


@lru_cache(maxsize = None)
def get_slot_source(slot_index : int) -> Tuple[Optional[object], Optional[numpy.ndarray]]:
	"""The averaged source face for a slot, and the first source frame (some swapper models want it)."""
	from facefusion.processors.modules.face_swapper.core import extract_source_face
	from facefusion.vision import read_static_images

	source_vision_frames = read_static_images(get_slots()[slot_index].get('sources') or [])
	source_face = extract_source_face(source_vision_frames)

	if not source_face:
		emit(f'WARNING Face {slot_index + 1}: no face found in its source images; that character is left as it is')
	return source_face, (source_vision_frames[0] if source_vision_frames else None)


def get_slot_gender(slot_index : int) -> Optional[str]:
	gender = get_slots()[slot_index].get('gender') or 'auto'

	if gender in ('male', 'female'):
		return gender
	source_face = get_slot_source(slot_index)[0]
	return getattr(source_face, 'gender', None) if source_face else None


def calculate_distance(face, embeddings : List[numpy.ndarray]) -> float:
	return min(1.0 - float(numpy.dot(face.embedding_norm, embedding)) for embedding in embeddings)


def get_slot_embeddings(slot_index : int) -> List[numpy.ndarray]:
	return list(get_reference_embeddings(slot_index)) or _locked.get(slot_index) or []


def load_lock_file() -> None:
	global _lock_file_loaded

	if _lock_file_loaded:
		return
	_lock_file_loaded = True
	lock_file = get_spec().get('lock_file')

	if not lock_file or not os.path.isfile(lock_file):
		return
	try:
		with open(lock_file, encoding = 'utf-8') as handle:
			saved = json.load(handle).get('slots') or {}
	except (OSError, ValueError):
		return
	for key, entry in saved.items():
		slot_index = int(key)

		if slot_index < len(get_slots()) and not get_reference_embeddings(slot_index) and entry.get('embedding'):
			_locked[slot_index] = [ normalize(entry.get('embedding')) ]
			emit(f'Face {slot_index + 1} follows the character locked in an earlier pass ({entry.get("crop") or lock_file})')


def save_crop(vision_frame : numpy.ndarray, face, path : str) -> bool:
	from facefusion.vision import write_image

	height, width = vision_frame.shape[:2]
	x1, y1, x2, y2 = [ float(value) for value in face.bounding_box ]
	side = max(x2 - x1, y2 - y1) * 1.8
	cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
	left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
	right, bottom = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))

	if right - left < 8 or bottom - top < 8:
		return False
	return bool(write_image(path, vision_frame[top:bottom, left:right]))


def write_lock_file(new_locks : Dict[int, object], vision_frame : numpy.ndarray) -> None:
	lock_file = get_spec().get('lock_file')

	if not lock_file:
		return
	os.makedirs(os.path.dirname(lock_file), exist_ok = True)
	saved = {}

	if os.path.isfile(lock_file):
		try:
			with open(lock_file, encoding = 'utf-8') as handle:
				saved = json.load(handle).get('slots') or {}
		except (OSError, ValueError):
			saved = {}
	for slot_index, face in new_locks.items():
		crop_path = os.path.splitext(lock_file)[0] + f'_face{slot_index + 1}.png'
		entry = { 'embedding': [ float(value) for value in face.embedding_norm ] }

		if vision_frame is not None and save_crop(vision_frame, face, crop_path):
			entry['crop'] = crop_path
		saved[str(slot_index)] = entry
		age = f', age {face.age.start}-{face.age.stop}' if isinstance(face.age, range) else ''
		emit(f'Face {slot_index + 1} locked onto a {face.gender or "unknown"} character{age}' + (f' - see {entry["crop"]}' if entry.get('crop') else ''))
	temp_path = lock_file + '.tmp' + str(os.getpid())

	with open(temp_path, 'w', encoding = 'utf-8') as handle:
		json.dump({ 'slots': saved }, handle)
	os.replace(temp_path, lock_file)


def ensure_locks(target_faces : List, vision_frame : numpy.ndarray) -> None:
	"""Lock every slot that has neither references nor a lock yet onto a character in this frame."""
	from facefusion.face_selector import sort_faces_by_order
	from facefusion import state_manager

	pending = [ slot_index for slot_index in range(len(get_slots())) if not get_slot_embeddings(slot_index) ]

	if not pending:
		return
	with _lock:
		load_lock_file()
		pending = [ slot_index for slot_index in pending if not get_slot_embeddings(slot_index) ]

		if not pending:
			return
		distance_limit = float(get_spec().get('distance', 0.65))
		known = [ get_slot_embeddings(slot_index) for slot_index in range(len(get_slots())) ]
		known = [ embeddings for embeddings in known if embeddings ]
		# A face that is already somebody's character cannot be locked onto again.
		candidates = [ face for face in sort_faces_by_order(target_faces, get_spec().get('order') or state_manager.get_item('face_selector_order') or 'large-small') if all(calculate_distance(face, embeddings) > distance_limit for embeddings in known) ]
		new_locks = {}

		for slot_index in pending:
			gender = get_slot_gender(slot_index)
			matches = [ face for face in candidates if not gender or face.gender == gender ]

			# With two faces the user asked for two characters, and the gender
			# classifier is wrong often enough on small or turned faces that a
			# strict filter can leave Face 2 with nobody at all - so there it is
			# a preference. A lone face, or a gender pinned in Settings, stays
			# strict, as upstream: a female source only replaces a female face.
			if not matches and len(get_slots()) > 1 and (get_slots()[slot_index].get('gender') or 'auto') == 'auto':
				matches = candidates

			if matches:
				face = matches[0]
				candidates.remove(face)
				_locked[slot_index] = [ normalize(face.embedding_norm) ]
				new_locks[slot_index] = face
		if new_locks:
			write_lock_file(new_locks, vision_frame)


def assign_faces(target_faces : List, vision_frame : numpy.ndarray) -> List[Tuple[int, object]]:
	"""(slot index, face) for every face that belongs to a slot's character, one-to-one."""
	if not target_faces:
		return []
	ensure_locks(target_faces, vision_frame)
	distance_limit = float(get_spec().get('distance', 0.65))
	pairs = []

	for slot_index in range(len(get_slots())):
		embeddings = get_slot_embeddings(slot_index)

		if not embeddings:
			continue
		for face_index, face in enumerate(target_faces):
			distance = calculate_distance(face, embeddings)

			if distance <= distance_limit:
				pairs.append((distance, slot_index, face_index))
	pairs.sort()
	used_slots = set()
	used_faces = set()
	assigned = []

	for _, slot_index, face_index in pairs:
		if slot_index in used_slots or face_index in used_faces:
			continue
		used_slots.add(slot_index)
		used_faces.add(face_index)
		assigned.append((slot_index, target_faces[face_index]))
	return assigned


def select_slot_faces(target_vision_frames : List[numpy.ndarray]) -> List[Tuple[int, object]]:
	"""The same target faces upstream select_faces would consider, assigned to slots."""
	from facefusion import state_manager
	from facefusion.common_helper import get_middle
	from facefusion.face_creator import get_static_faces
	from facefusion.face_tracker import track_faces

	if state_manager.get_item('face_tracker_score') > 0:
		target_faces = track_faces(target_vision_frames, state_manager.get_item('face_tracker_score'))
	else:
		target_faces = get_static_faces([ get_middle(target_vision_frames) ])
	return assign_faces(target_faces, get_middle(target_vision_frames))
