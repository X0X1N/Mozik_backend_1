import os
import cv2
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core import gaussian_blur_region, expand_box
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding

YOLOV5_WEIGHTS = os.path.join(os.path.dirname(__file__), "best.pt")
DETECT_THUMB_W = 640
FACE_PAD = 0.05

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")
DEFAULT_EMOJI_FILES = [
    "emoji1.png",
    "emoji2.png",
    "emoji3.png",
    "emoji4.png",
    "emoji5.png",
]

# ==========================================
# Dataclasses
# ==========================================
@dataclass
class UBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class DBox(UBox):
    enabled: bool = True
    identity: Optional[str] = None
    emb: Optional[np.ndarray] = None


# ==========================================
# Load YOLO
# ==========================================
try:
    DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
except Exception as e:
    print(f"[ERROR] YOLOv5 load error: {e}")
    DETECTOR = None


# ==========================================
# Face Detect (YOLO + Haar Fallback)
# ==========================================
def _detect_on_frame(detector, frame) -> List[DBox]:
    """
    1차: YOLOv5
    2차: 결과가 없으면 HaarCascade
    """
    h, w = frame.shape[:2]
    faces: List[DBox] = []

    # ---------- 1단계: YOLO ----------
    if detector is not None:
        if w > DETECT_THUMB_W:
            scale = DETECT_THUMB_W / float(w)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        else:
            scale = 1.0
            small = frame

        try:
            rects_small = detect_faces_yolov5(detector, small)
        except Exception as e:
            print(f"[WARN] YOLO detect 실패: {e}")
            rects_small = []

        pad = float(FACE_PAD)

        for (x, y, w0, h0) in rects_small:
            if scale != 1.0:
                x, y, w0, h0 = int(x / scale), int(y / scale), int(w0 / scale), int(h0 / scale)

            ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)

            if ew >= 12 and eh >= 12:
                faces.append(DBox(ex, ey, ew, eh, True))

    # ---------- 2단계: YOLO 결과 없으면 Haar ----------
    if not faces:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            rects = haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            pad = float(FACE_PAD)
            for (x, y, w0, h0) in rects:
                ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)
                if ew >= 12 and eh >= 12:
                    faces.append(DBox(ex, ey, ew, eh, True))
        except Exception as e:
            print(f"[WARN] Haar detect 실패: {e}")

    return faces


# ==========================================
# ArcFace Matching
# ==========================================
def _annotate_authorized_faces(registry: FaceRegistry, faces: List[DBox], frame_bgr):
    if not faces:
        return

    h, w = frame_bgr.shape[:2]

    for b in faces:
        if b.identity is not None:
            continue

        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(w, b.x + b.w), min(h, b.y + b.h)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            if b.emb is None:
                b.emb = get_embedding(crop)

            name, dist = registry.match(b.emb)
            b.identity = name

        except Exception:
            b.identity = None


# ==========================================
# Policy System (exclude / force / normal)
# ==========================================
def _should_blur_auto_face(registry: FaceRegistry, box: DBox, auth_exclude_enable: bool) -> bool:
    if not box.enabled:
        return False

    if auth_exclude_enable and box.identity:
        mode = registry.get_policy(box.identity)

        if mode == "exclude":
            return False

        if mode == "force":
            return True

    return True


# ==========================================
# Mosaic
# ==========================================
def _apply_mosaic(frame, x, y, w, h, mosaic_size=15):
    h_img, w_img = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    small_w = max(1, (x2 - x1) // mosaic_size)
    small_h = max(1, (y2 - y1) // mosaic_size)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = mosaic


# ==========================================
# Emoji Overlay
# ==========================================
def _load_emoji_image(emoji_path: Optional[str], emoji_index: int = 0):
    if emoji_path:
        path = emoji_path
    else:
        idx = emoji_index if 0 <= emoji_index < len(DEFAULT_EMOJI_FILES) else 0
        path = os.path.join(EMOJI_DIR, DEFAULT_EMOJI_FILES[idx])

    if not os.path.exists(path):
        print("[emoji] file not found:", path)
        return None

    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def _overlay_emoji(frame, x, y, w, h, emoji_img):
    if emoji_img is None:
        return

    h_img, w_img = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 <= x1 or y2 <= y1:
        return

    tw, th = x2 - x1, y2 - y1

    emoji_resized = cv2.resize(emoji_img, (tw, th), interpolation=cv2.INTER_AREA)

    if emoji_resized.shape[2] == 4:
        b, g, r, a = cv2.split(emoji_resized)
        emoji_rgb = cv2.merge((b, g, r))
        alpha = (a / 255.0)[..., None]

        roi = frame[y1:y2, x1:x2].astype(float)
        blended = alpha * emoji_rgb + (1 - alpha) * roi

        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized


# ==========================================
# Main Function — Process Video
# ==========================================
def process_video(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
):
    # DETECTOR 가 None 이어도 Haar fallback 으로 탐지 시도한다.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"비디오 파일을 열 수 없음: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise IOError(f"비디오 writer 생성 실패: {output_path}")

    registry = FaceRegistry()

    emoji_img = None
    if style == "emoji":
        emoji_img = _load_emoji_image(emoji_path, emoji_index)
        if emoji_img is None:
            style = "blur"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = _detect_on_frame(DETECTOR, frame)

            if auth_exclude_enable and faces:
                _annotate_authorized_faces(registry, faces, frame)

            for box in faces:
                if not _should_blur_auto_face(registry, box, auth_exclude_enable):
                    continue

                if style == "blur":
                    gaussian_blur_region(frame, box.x, box.y, box.w, box.h, blur_strength)

                elif style == "mosaic":
                    _apply_mosaic(
                        frame,
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        mosaic_size=max(8, min(30, blur_strength)),
                    )

                elif style == "emoji":
                    _overlay_emoji(frame, box.x, box.y, box.w, box.h, emoji_img)

                else:
                    gaussian_blur_region(frame, box.x, box.y, box.w, box.h, blur_strength)

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
