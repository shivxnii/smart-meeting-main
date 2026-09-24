# ============================================================
# SMART MEETING - SPEAKER ENGINE
# ============================================================
# Changes in this version:
#   1. Lost-track recovery: when a face track disappears (head turn,
#      detector miss, leaving the frame) it is remembered for a while.
#      When a face shows up again it gets its OLD Person ID back
#      instead of a brand new one (Person 8 -> 9 -> 10 problem).
#   2. Tracks are kept alive longer before being dropped.
#   3. The blur-other-people block was pasted 3 times; now only once.
#   4. Newly created tracks are no longer marked "missed" instantly.
#   5. VERSION 3: MAX_PEOPLE cap, stale-track re-acquire, [ID] log lines.
#      (If you can read this line, you have the latest file.)
# ============================================================

import time
import threading
from collections import deque

import cv2
import numpy as np

from smart_meeting.config import (
    ASPECT_RATIO,
    FRAMING_PADDING_FACTOR,
    FRAMING_SMOOTH_ALPHA,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    TALK_THRESHOLD,
    FACE_DET_CONF,
    FACE_DET_SCALE,
    FACE_DET_EVERY_N_FRAMES,
    IOU_TRACK_THRESH,
    MAX_FAILED_DET,
    WINDOW_FRAMES,
    WINDOW_SECONDS,
    SPEAKER_SWITCH_HYSTERESIS,
    SPEAKER_RELEASE_HYSTERESIS,
    TALKNET_FACE_SIZE,
    MIN_FACE_SIZE,
)

import smart_meeting.config as _cfg

# Audio/video alignment settings (optional keys in config.py).
ALIGN_AV_WINDOW = getattr(_cfg, "ALIGN_AV_WINDOW", True)
AV_FPS = getattr(_cfg, "AV_FPS", 25)
AV_WINDOW_FRAMES = getattr(_cfg, "AV_WINDOW_FRAMES", 25)
AV_MAX_GAP_SEC = getattr(_cfg, "AV_MAX_GAP_SEC", 0.30)
AV_MIN_COVERAGE = getattr(_cfg, "AV_MIN_COVERAGE", 0.70)
DEBUG_TALKNET = getattr(_cfg, "DEBUG_TALKNET", False)
TALKNET_REFERENCE_CROP = getattr(_cfg, "TALKNET_REFERENCE_CROP", True)

# Identity persistence settings (optional keys in config.py).
# How long (seconds) a disappeared person's ID is remembered.
LOST_TRACK_KEEP_SEC = getattr(_cfg, "LOST_TRACK_KEEP_SEC", 120.0)
# Max distance (in face-sizes) between the old and new face position
# for the old ID to be reused.
LOST_MATCH_MAX_DIST = getattr(_cfg, "LOST_MATCH_MAX_DIST", 4.0)
# If nobody is visible and exactly one person is remembered, reuse that
# ID whatever the position (good for single-person use).
LOST_SOLO_REUSE = getattr(_cfg, "LOST_SOLO_REUSE", True)
# Hard cap on simultaneous people. 0 = unlimited (normal meeting).
# Set MAX_PEOPLE = 1 in config.py when only one person is in front of
# the camera: then that person can never receive a second ID.
MAX_PEOPLE = int(getattr(_cfg, "MAX_PEOPLE", 0))
# A visible track counts as "briefly lost" after this many missed frames.
STALE_MIN_MISSED = int(getattr(_cfg, "STALE_MIN_MISSED", 3))
# A track survives at least this many missed frames (~10 fps => 4 sec).
TRACK_PATIENCE = max(int(MAX_FAILED_DET), int(getattr(_cfg, "TRACK_PATIENCE", 40)))

from smart_meeting.talknet_cpu import (
    TalkNetASD,
    S3FDDetector,
)

from smart_meeting.subtitle_engine import (
    SubtitleEngine,
)


# ============================================================
# IOU
# ============================================================

def compute_iou(box_a, box_b):

    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    if intersection <= 0:
        return 0.0

    area_a = (
        max(1.0, float(box_a[2]) - float(box_a[0]))
        * max(1.0, float(box_a[3]) - float(box_a[1]))
    )

    area_b = (
        max(1.0, float(box_b[2]) - float(box_b[0]))
        * max(1.0, float(box_b[3]) - float(box_b[1]))
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(box_a, box_b):

    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0

    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


# ============================================================
# PERSON TRACK
# ============================================================

class PersonTrack:

    _next_id = 1

    def __init__(self, box, track_id=None):

        # track_id is given when an old (lost) identity is revived.
        if track_id is None:
            self.id = PersonTrack._next_id
            PersonTrack._next_id += 1
        else:
            self.id = track_id

        self.box = list(box)

        self.smooth_box = list(box)

        self.missed_frames = 0

        self.age = 1

        self.visible_frames = 1

        self.face_history = deque(
            maxlen=max(WINDOW_FRAMES * 3, 60)
        )

        self.last_score = -10.0

        self.is_speaking = False

        self.last_seen = time.time()

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self, box, face_crop, timestamp):

        self.missed_frames = 0

        self.age += 1

        self.visible_frames += 1

        self.last_seen = timestamp

        # Smoother than directly replacing the box every frame.
        alpha = 0.45

        for i in range(4):
            self.smooth_box[i] = (
                (1.0 - alpha) * self.smooth_box[i]
                + alpha * float(box[i])
            )

        self.box = list(box)

        if face_crop is not None:
            self.face_history.append((timestamp, face_crop))

    # --------------------------------------------------------
    # MISSED
    # --------------------------------------------------------

    def mark_missed(self):

        self.missed_frames += 1

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    @property
    def center(self):

        return (
            (self.smooth_box[0] + self.smooth_box[2]) / 2.0,
            (self.smooth_box[1] + self.smooth_box[3]) / 2.0,
        )

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    @property
    def size(self):

        width = max(1.0, self.smooth_box[2] - self.smooth_box[0])
        height = max(1.0, self.smooth_box[3] - self.smooth_box[1])

        return max(width, height)

    # --------------------------------------------------------
    # FACE WINDOW
    # --------------------------------------------------------

    def get_faces_for_window(self, end_time):

        start_time = end_time - WINDOW_SECONDS

        selected = []

        for timestamp, crop in self.face_history:
            if start_time <= timestamp <= end_time:
                selected.append(crop)

        if len(selected) > WINDOW_FRAMES:
            selected = selected[-WINDOW_FRAMES:]

        return selected


# ============================================================
# SMART MEETING DIRECTOR
# ============================================================

class SmartMeetingDirector:

    def __init__(self):

        print("[Director] Loading AI models...")

        self.face_detector = S3FDDetector()

        self.talknet = TalkNetASD()

        self.subtitle_engine = SubtitleEngine(model_name="base")

        self.tracks = []

        # Tracks that disappeared recently. Their IDs can be reused.
        self.lost_tracks = []

        self.active_speaker_id = None

        self.active_speaker_score = -10.0

        self.speaker_switch_counter = 0

        self.speaker_release_counter = 0

        self.frame_counter = 0

        self.last_detection_boxes = []

        self.last_detection_time = 0.0

        self.person_count = 0

        self.cam_cx = None
        self.cam_cy = None
        self.cam_half_w = None

        self.lock = threading.Lock()

        print("[Director] Ready.")

    # ========================================================
    # FILTER DETECTIONS
    # ========================================================

    def _filter_detections(self, detections, frame_shape):

        height, width = frame_shape[:2]

        boxes = []

        for detection in detections:

            if len(detection) < 4:
                continue

            try:
                x1 = float(detection[0])
                y1 = float(detection[1])
                x2 = float(detection[2])
                y2 = float(detection[3])

                confidence = (
                    float(detection[4])
                    if len(detection) >= 5
                    else 1.0
                )
            except Exception:
                continue

            if confidence < FACE_DET_CONF:
                continue

            # Clamp.
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                continue

            boxes.append([x1, y1, x2, y2])

        # Remove duplicate/overlapping detections.
        if len(boxes) <= 1:
            return boxes

        keep = []

        # Largest boxes first.
        boxes.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )

        for box in boxes:

            duplicate = False

            for existing in keep:
                if compute_iou(box, existing) >= 0.55:
                    duplicate = True
                    break

            if not duplicate:
                keep.append(box)

        return keep

    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces_in_frame(self, frame_bgr):

        should_detect = (
            self.frame_counter
            % max(1, FACE_DET_EVERY_N_FRAMES)
            == 0
        )

        # Reuse the previous detections between S3FD passes.
        if not should_detect and self.last_detection_boxes:
            return list(self.last_detection_boxes)

        try:

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detections = self.face_detector.detect_faces(
                rgb,
                conf_th=FACE_DET_CONF,
                scales=(FACE_DET_SCALE,),
            )

            boxes = self._filter_detections(
                detections,
                frame_bgr.shape,
            )

            self.last_detection_boxes = boxes

            self.last_detection_time = time.time()

            return list(boxes)

        except Exception as exc:

            print("[Director] S3FD error:", repr(exc))

            return []

    # ========================================================
    # FACE CROP
    # ========================================================

    def _extract_face_crop(self, frame_bgr, box):
        """
        Face crop identical to the original TalkNet demo (demoTalkNet.py):
        a box of 2.8 x half-size around the face (cropScale 0.4), resized to
        224, converted to gray, then the central 112 x 112 is used. That is
        a tight, slightly-below-centre crop that keeps the mouth large.
        """

        if not TALKNET_REFERENCE_CROP:
            return self._extract_face_crop_legacy(frame_bgr, box)

        height, width = frame_bgr.shape[:2]

        x1, y1, x2, y2 = [float(v) for v in box]

        bs = max(x2 - x1, y2 - y1) / 2.0

        if bs < 4.0:
            return None

        cs = 0.40

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        top = int(np.floor(cy - bs))
        bottom = int(np.floor(cy + bs * (1 + 2 * cs)))
        left = int(np.floor(cx - bs * (1 + cs)))
        right = int(np.floor(cx + bs * (1 + cs)))

        pad_top = max(0, -top)
        pad_left = max(0, -left)
        pad_bottom = max(0, bottom - height)
        pad_right = max(0, right - width)

        crop = frame_bgr[
            max(top, 0):min(bottom, height),
            max(left, 0):min(right, width),
        ]

        if crop.size == 0:
            return None

        if pad_top or pad_left or pad_bottom or pad_right:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(110, 110, 110),
            )

        crop = cv2.resize(crop, (224, 224))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray = gray[56:168, 56:168]

        if gray.shape[0] != TALKNET_FACE_SIZE:
            gray = cv2.resize(
                gray,
                (TALKNET_FACE_SIZE, TALKNET_FACE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        return gray

    def _extract_face_crop_legacy(self, frame_bgr, box):

        height, width = frame_bgr.shape[:2]

        x1, y1, x2, y2 = [int(round(v)) for v in box]

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        face_width = x2 - x1
        face_height = y2 - y1

        size = max(face_width, face_height)

        # More stable context around face.
        pad = int(size * 0.30)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        px1 = max(0, cx - size // 2 - pad)
        py1 = max(0, cy - size // 2 - pad)
        px2 = min(width, cx + size // 2 + pad)
        py2 = min(height, cy + size // 2 + pad)

        if px2 <= px1 or py2 <= py1:
            return None

        crop = frame_bgr[py1:py2, px1:px2]

        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            (TALKNET_FACE_SIZE, TALKNET_FACE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        return gray

    # ========================================================
    # TRACK MATCHING
    # ========================================================

    def _match_detection_to_track(self, track, box):

        iou = compute_iou(track.smooth_box, box)

        if iou >= IOU_TRACK_THRESH:
            return True

        # If IoU becomes temporarily small because the face detector
        # shifts slightly, allow a center-distance based match.
        distance = center_distance(track.smooth_box, box)

        track_size = max(1.0, track.size)

        if distance <= track_size * 0.85:
            return True

        return False

    # ========================================================
    # LOST TRACK RECOVERY
    # ========================================================

    def _recover_lost_id(self, box, timestamp):
        """
        Called just before a brand new Person ID would be created.
        Returns the ID of a recently lost track if the new face plausibly
        belongs to it, otherwise None. The returned track is removed from
        the lost list.
        """

        if not self.lost_tracks:
            return None

        # Nobody is currently visible: whichever remembered person was
        # seen most recently is overwhelmingly likely to be the same
        # person reappearing (this was previously restricted to "exactly
        # one remembered person", which stopped working once a few ID
        # splits had already happened in the same session and multiple
        # old IDs were being remembered at once).
        if LOST_SOLO_REUSE and not self.tracks:
            newest = max(self.lost_tracks, key=lambda t: t.last_seen)
            self.lost_tracks.remove(newest)
            print(
                f"[ID] Person {newest.id} re-acquired "
                "(nobody else visible, reusing most recent ID)"
            )
            return newest.id

        best = None
        best_score = 1e9

        for lost in self.lost_tracks:

            dist = (
                center_distance(lost.smooth_box, box)
                / max(lost.size, 1.0)
            )

            if dist > LOST_MATCH_MAX_DIST:
                continue

            age = max(0.0, timestamp - lost.last_seen)

            # Closer and more recently seen is better.
            score = dist + 0.02 * age

            if score < best_score:
                best = lost
                best_score = score

        if best is None:
            if self.lost_tracks:
                print(
                    "[ID] No lost track close enough to reuse "
                    f"(closest was beyond {LOST_MATCH_MAX_DIST}x face size) "
                    "-> creating new ID"
                )
            return None

        self.lost_tracks.remove(best)

        return best.id

    # ========================================================
    # TRACK UPDATE
    # ========================================================

    def update_tracks(self, frame_bgr, detected_boxes, timestamp):

        with self.lock:

            matched_tracks = set()

            matched_boxes = set()

            candidates = []

            # ------------------------------------------------
            # Build matching candidates.
            # ------------------------------------------------

            for track_index, track in enumerate(self.tracks):

                for box_index, box in enumerate(detected_boxes):

                    iou = compute_iou(track.smooth_box, box)

                    distance = center_distance(track.smooth_box, box)

                    normalized_distance = (
                        distance / max(track.size, 1.0)
                    )

                    # Strong IoU gets priority.
                    if iou >= IOU_TRACK_THRESH:

                        score = iou + 0.15 * max(
                            0.0,
                            1.0 - normalized_distance,
                        )

                        candidates.append(
                            (score, track_index, box_index)
                        )

                    # Loose center-distance recovery.
                    elif normalized_distance <= 0.70:

                        score = 0.35 - normalized_distance * 0.20

                        candidates.append(
                            (score, track_index, box_index)
                        )

            candidates.sort(reverse=True)

            # ------------------------------------------------
            # Greedy assignment.
            # ------------------------------------------------

            for score, track_index, box_index in candidates:

                if track_index in matched_tracks:
                    continue

                if box_index in matched_boxes:
                    continue

                track = self.tracks[track_index]

                box = detected_boxes[box_index]

                if not self._match_detection_to_track(track, box):
                    continue

                crop = self._extract_face_crop(frame_bgr, box)

                track.update(box, crop, timestamp)

                matched_tracks.add(track_index)

                matched_boxes.add(box_index)

            # Tracks that exist before new ones are created below.
            existing_count = len(self.tracks)

            # ------------------------------------------------
            # Create tracks only for genuinely unmatched
            # detections. An old ID is reused when possible.
            # ------------------------------------------------

            for box_index, box in enumerate(detected_boxes):

                if box_index in matched_boxes:
                    continue

                crop = self._extract_face_crop(frame_bgr, box)

                # ----------------------------------------------
                # Step 1: is this box actually the SAME PERSON as
                # an existing track — matched already this frame,
                # or not? A loose IoU/center-distance check,
                # regardless of how long that track has gone
                # unmatched. This is the key fix: previously a
                # "near an existing track" box was just DROPPED
                # (never refreshing that track), so the track kept
                # aging with a stale position until it exceeded
                # patience and a brand new ID got created for the
                # same person. Now it directly refreshes the
                # closest matching track instead.
                # ----------------------------------------------

                nearest_index = None
                nearest_score = -1.0

                for track_index, track in enumerate(self.tracks):

                    iou = compute_iou(track.smooth_box, box)

                    distance = center_distance(track.smooth_box, box)

                    normalized_distance = distance / max(track.size, 1.0)

                    close = (
                        iou >= 0.20
                        or distance <= max(track.size * 0.50, 30.0)
                    )

                    if not close:
                        continue

                    score = iou - normalized_distance * 0.05

                    if score > nearest_score:
                        nearest_score = score
                        nearest_index = track_index

                if nearest_index is not None:

                    if nearest_index in matched_tracks:
                        # Two detected boxes both landed on a track
                        # already updated this frame: a genuine
                        # duplicate detection, not a new person.
                        continue

                    track = self.tracks[nearest_index]

                    track.smooth_box = list(box)

                    track.update(box, crop, timestamp)

                    matched_tracks.add(nearest_index)

                    matched_boxes.add(box_index)

                    print(
                        f"[ID] Person {track.id} re-acquired "
                        "(loose match, same person / face drifted)"
                    )

                    continue

                # ----------------------------------------------
                # Step 2: MAX_PEOPLE cap reached and nobody nearby
                # claimed this box. With a hard cap, a genuinely
                # new face cannot be a new person -> hand it to
                # whichever tracked person hasn't been matched yet
                # this frame (they likely moved further than
                # expected), or drop it as a false detection if
                # everyone is already accounted for.
                # ----------------------------------------------

                capped = bool(MAX_PEOPLE) and len(self.tracks) >= MAX_PEOPLE

                if capped:

                    unmatched_existing = [
                        i for i in range(len(self.tracks))
                        if i not in matched_tracks
                    ]

                    if unmatched_existing:

                        pick = max(
                            unmatched_existing,
                            key=lambda i: self.tracks[i].last_seen,
                        )

                        track = self.tracks[pick]

                        track.smooth_box = list(box)

                        track.update(box, crop, timestamp)

                        matched_tracks.add(pick)

                        matched_boxes.add(box_index)

                        print(
                            f"[ID] Person {track.id} re-acquired "
                            "(MAX_PEOPLE cap, same person assumed)"
                        )

                    continue

                # ----------------------------------------------
                # Step 3: genuinely new face, nowhere near any
                # existing track -> new (or revived) ID.
                # ----------------------------------------------

                revived_id = self._recover_lost_id(box, timestamp)

                track = PersonTrack(box, track_id=revived_id)

                track.update(box, crop, timestamp)

                self.tracks.append(track)

                print(
                    f"[ID] Person {track.id} "
                    f"{'REVIVED (old id reused)' if revived_id is not None else 'NEW'}"
                    f" | visible={len(self.tracks)} lost={len(self.lost_tracks)}"
                )

            # ------------------------------------------------
            # Handle tracks not detected this pass.
            # ------------------------------------------------

            surviving = []

            for index, track in enumerate(self.tracks):

                # Tracks created in this very call have
                # index >= existing_count and are not "missed".
                if index < existing_count and index not in matched_tracks:
                    track.mark_missed()

                if track.missed_frames <= TRACK_PATIENCE:
                    surviving.append(track)
                else:
                    # Remember it so the ID can be reused later.
                    self.lost_tracks.append(track)

            self.tracks = surviving

            # Forget people who have been gone for too long.
            self.lost_tracks = [
                t for t in self.lost_tracks
                if timestamp - t.last_seen <= LOST_TRACK_KEEP_SEC
            ]

            # ------------------------------------------------
            # Clean invalid active speaker.
            # ------------------------------------------------

            valid_ids = {track.id for track in self.tracks}

            if self.active_speaker_id not in valid_ids:
                self.active_speaker_id = None
                self.active_speaker_score = -10.0

            self.person_count = len(self.tracks)

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame_bgr, timestamp=None):

        if timestamp is None:
            timestamp = time.time()

        self.frame_counter += 1

        boxes = self.detect_faces_in_frame(frame_bgr)

        self.update_tracks(frame_bgr, boxes, timestamp)

    # ========================================================
    # AUDIO / VIDEO ALIGNMENT
    # ========================================================

    def _get_aligned_faces(self, track, end_time):
        """
        TalkNet expects 25 video frames per second and audio that covers
        exactly the same moment. Frames here arrive at the camera rate
        (about 10 fps) and at uneven times, so resample them onto a
        regular 25 fps grid ending at end_time (nearest frame per slot).

        Returns [] when the track does not have enough recent frames.
        """

        if not ALIGN_AV_WINDOW:
            return track.get_faces_for_window(end_time)

        history = list(track.face_history)

        if len(history) < 2:
            return []

        stamps = np.fromiter(
            (item[0] for item in history),
            dtype=np.float64,
            count=len(history),
        )

        n = int(AV_WINDOW_FRAMES)

        grid = end_time - (
            (n - 1 - np.arange(n)) / float(AV_FPS)
        )

        idx = np.searchsorted(stamps, grid)
        idx = np.clip(idx, 1, len(stamps) - 1)

        left = stamps[idx - 1]
        right = stamps[idx]

        idx = np.where(
            np.abs(grid - left) <= np.abs(right - grid),
            idx - 1,
            idx,
        )

        error = np.abs(stamps[idx] - grid)

        if float(np.mean(error <= AV_MAX_GAP_SEC)) < AV_MIN_COVERAGE:
            return []

        return [history[int(i)][1] for i in idx]

    def _align_audio(self, audio_data, sample_rate):
        """
        Keep only the most recent audio that matches the video window,
        instead of the older start of a longer buffer.
        """

        if not ALIGN_AV_WINDOW:
            return audio_data

        needed = int(
            round(
                sample_rate
                * (AV_WINDOW_FRAMES / float(AV_FPS))
            )
        )

        if len(audio_data) > needed:
            return audio_data[-needed:]

        return audio_data

    # ========================================================
    # ACTIVE SPEAKER
    # ========================================================

    def evaluate_active_speaker(
        self,
        audio_data,
        sample_rate=16000,
        window_end_time=None,
    ):

        if window_end_time is None:
            window_end_time = time.time()

        with self.lock:
            tracks_snapshot = list(self.tracks)

        if (
            not tracks_snapshot
            or audio_data is None
            or len(audio_data) == 0
        ):
            return (self.active_speaker_id, -10.0)

        audio_data = self._align_audio(audio_data, sample_rate)

        best_id = None

        best_score = -10.0

        # ----------------------------------------------------
        # Evaluate each tracked face.
        # ----------------------------------------------------

        for track in tracks_snapshot:

            faces = self._get_aligned_faces(track, window_end_time)

            required_frames = max(5, int(WINDOW_FRAMES * 0.25))

            if DEBUG_TALKNET:
                print(
                    "[TalkNet-DEBUG] track",
                    track.id,
                    "faces_available=",
                    len(faces),
                    "required=",
                    required_frames,
                )

            if len(faces) < required_frames:
                continue

            if len(faces) > WINDOW_FRAMES:

                indices = np.linspace(
                    0,
                    len(faces) - 1,
                    WINDOW_FRAMES,
                ).astype(int)

                faces = [faces[index] for index in indices]

            try:

                score, _ = self.talknet.compute_talking_scores(
                    audio_data,
                    sample_rate,
                    faces,
                )

            except Exception as exc:

                print(
                    "[TalkNet] Track " f"{track.id} error:",
                    repr(exc),
                )

                continue

            track.last_score = float(score)

            if DEBUG_TALKNET:
                print(
                    f"[SCORE] track {track.id} "
                    f"score={float(score):.2f} "
                    f"threshold={TALK_THRESHOLD}"
                )

            if score > best_score:
                best_score = float(score)
                best_id = track.id

        with self.lock:

            self.active_speaker_score = best_score

            # ------------------------------------------------
            # No candidate, or no confident speech.
            # ------------------------------------------------

            if best_id is None or best_score <= TALK_THRESHOLD:

                self.speaker_release_counter += 1

                self.speaker_switch_counter = 0

                if (
                    self.speaker_release_counter
                    >= SPEAKER_RELEASE_HYSTERESIS
                ):
                    self.active_speaker_id = None

                for track in self.tracks:
                    track.is_speaking = False

                return (self.active_speaker_id, best_score)

            # ------------------------------------------------
            # Confident speech.
            # ------------------------------------------------

            self.speaker_release_counter = 0

            if best_id == self.active_speaker_id:

                self.speaker_switch_counter = 0

            else:

                self.speaker_switch_counter += 1

                if (
                    self.speaker_switch_counter
                    >= SPEAKER_SWITCH_HYSTERESIS
                ):
                    self.active_speaker_id = best_id
                    self.speaker_switch_counter = 0

            for track in self.tracks:
                track.is_speaking = (
                    track.id == self.active_speaker_id
                )

            return (self.active_speaker_id, best_score)

    # ========================================================
    # FALLBACK SPEAKER (single-person voice-activity credit)
    # ========================================================

    def set_fallback_speaker(self, speaker_id):
        """
        Used when app.py's SINGLE_PERSON_FALLBACK credits someone as
        speaking from voice activity alone, because TalkNet didn't have
        enough synced face frames yet to score them itself. Without this,
        the transcript/status correctly shows who is talking but the
        camera crop never zooms in, because it only reacts to
        self.active_speaker_id.
        """

        with self.lock:

            valid_ids = {track.id for track in self.tracks}

            if speaker_id not in valid_ids:
                return

            self.active_speaker_id = speaker_id
            self.speaker_switch_counter = 0
            self.speaker_release_counter = 0

            for track in self.tracks:
                track.is_speaking = (track.id == speaker_id)

    # ========================================================
    # CENTERED SPEAKER VIEW
    # ========================================================

    def render_centered_speaker_frame(
        self,
        frame_bgr,
        subtitle_text="",
    ):

        height, width = frame_bgr.shape[:2]

        with self.lock:

            active_id = self.active_speaker_id

            active_track = None

            for track in self.tracks:
                if track.id == active_id:
                    active_track = track
                    break

        # ----------------------------------------------------
        # No active speaker.
        # ----------------------------------------------------

        if active_track is None:
            return self.render_room_overview_frame(
                frame_bgr,
                subtitle_text,
            )

        cx, cy = active_track.center

        person_size = active_track.size

        crop_height = max(
            120,
            int(person_size * FRAMING_PADDING_FACTOR),
        )

        crop_width = int(crop_height * ASPECT_RATIO)

        crop_width = min(width, crop_width)

        crop_height = min(height, crop_height)

        target_cx = cx

        target_cy = cy

        target_half_width = crop_width / 2.0

        if self.cam_cx is None:

            self.cam_cx = target_cx
            self.cam_cy = target_cy
            self.cam_half_w = target_half_width

        else:

            alpha = FRAMING_SMOOTH_ALPHA

            self.cam_cx = (1.0 - alpha) * self.cam_cx + alpha * target_cx
            self.cam_cy = (1.0 - alpha) * self.cam_cy + alpha * target_cy
            self.cam_half_w = (
                (1.0 - alpha) * self.cam_half_w
                + alpha * target_half_width
            )

        crop_width = int(self.cam_half_w * 2.0)

        crop_width = max(100, min(width, crop_width))

        crop_height = int(crop_width / ASPECT_RATIO)

        crop_height = max(100, min(height, crop_height))

        x1 = int(self.cam_cx - crop_width / 2.0)

        y1 = int(self.cam_cy - crop_height / 2.0)

        x1 = max(0, min(x1, width - crop_width))

        y1 = max(0, min(y1, height - crop_height))

        x2 = min(width, x1 + crop_width)

        y2 = min(height, y1 + crop_height)

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return self.render_room_overview_frame(
                frame_bgr,
                subtitle_text,
            )

        output = cv2.resize(
            crop,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.putText(
            output,
            "ACTIVE SPEAKER: " f"Person {active_id}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )

        output = SubtitleEngine.draw_subtitles_overlay(
            output,
            subtitle_text,
        )

        return output

    # ========================================================
    # ROOM OVERVIEW
    # ========================================================

    def render_room_overview_frame(
        self,
        frame_bgr,
        subtitle_text="",
    ):

        display = frame_bgr.copy()

        with self.lock:
            tracks = list(self.tracks)
            active_id = self.active_speaker_id

        # ----------------------------------------------------
        # Draw tracked faces.
        # ----------------------------------------------------

        active_box = None

        for track in tracks:

            x1, y1, x2, y2 = [
                int(round(v)) for v in track.smooth_box
            ]

            x1 = max(0, min(display.shape[1] - 1, x1))
            y1 = max(0, min(display.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(display.shape[1], x2))
            y2 = max(y1 + 1, min(display.shape[0], y2))

            active = track.id == active_id and track.is_speaking

            if active:
                active_box = (x1, y1, x2, y2)
                box_color = (0, 255, 100)
                thickness = 3
            else:
                box_color = (255, 180, 0)
                thickness = 2

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness,
            )

            label = f"Person {track.id}"

            if active:
                label += "  SPEAKING " f"{track.last_score:.2f}"

            cv2.putText(
                display,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                box_color,
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Room status.
        # ----------------------------------------------------

        cv2.putText(
            display,
            "ROOM: " f"{len(tracks)} PERSON(S)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Zoom into the active speaker: crop the frame around
        # their box (with padding) and drop everyone else.
        # If nobody is speaking, keep the full room view.
        # ----------------------------------------------------

        if active_box is not None:

            bx1, by1, bx2, by2 = active_box
            box_w = max(1, bx2 - bx1)
            box_h = max(1, by2 - by1)

            pad_x = int(box_w * 0.7)
            pad_top = int(box_h * 0.9)
            pad_bottom = int(box_h * 0.5)

            cx1 = max(0, bx1 - pad_x)
            cy1 = max(0, by1 - pad_top)
            cx2 = min(display.shape[1], bx2 + pad_x)
            cy2 = min(display.shape[0], by2 + pad_bottom)

            crop_w = max(1, cx2 - cx1)
            crop_h = max(1, cy2 - cy1)

            target_ratio = TARGET_WIDTH / float(TARGET_HEIGHT)
            crop_ratio = crop_w / float(crop_h)

            if crop_ratio > target_ratio:
                new_h = int(crop_w / target_ratio)
                extra = max(0, new_h - crop_h)
                cy1 = max(0, cy1 - extra // 2)
                cy2 = min(display.shape[0], cy1 + new_h)
                cy1 = max(0, cy2 - new_h)
            else:
                new_w = int(crop_h * target_ratio)
                extra = max(0, new_w - crop_w)
                cx1 = max(0, cx1 - extra // 2)
                cx2 = min(display.shape[1], cx1 + new_w)
                cx1 = max(0, cx2 - new_w)

            cropped = display[cy1:cy2, cx1:cx2]

            if cropped.size > 0:
                display = cropped

        display = cv2.resize(
            display,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        return SubtitleEngine.draw_subtitles_overlay(
            display,
            subtitle_text,
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status_summary(self):

        with self.lock:

            attendees = []

            for track in self.tracks:

                attendees.append(
                    {
                        "id": track.id,
                        "score": round(float(track.last_score), 3),
                        "is_speaking": bool(track.is_speaking),
                        "missed_frames": int(track.missed_frames),
                    }
                )

            return {
                "persons_detected": len(self.tracks),
                "active_speaker_id": self.active_speaker_id,
                "speaking_score": round(
                    float(self.active_speaker_score),
                    3,
                ),
                "is_speaking": self.active_speaker_id is not None,
                "attendees": attendees,
            }
