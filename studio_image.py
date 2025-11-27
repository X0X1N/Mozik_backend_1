import os
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core import gaussian_blur_region, expand_box
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding

# =============================
# 기본 설정
# =============================
YOLOV5_WEIGHTS = os.path.join(os.path.dirname(__file__), "best.pt")
DETECT_THUMB_W = 640
FACE_PAD = 0.05   # 얼굴 부분만 타이트하게 잡기 위한 패딩

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")
DEFAULT_EMOJI_FILES = ["emoji1.png", "emoji2.png", "emoji3.png", "emoji4.png", "emoji5.png"]


# =============================
# Dataclasses
# =============================
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


# =============================
# YOLO 모델 로드
# =============================
try:
    DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
    print("[YOLO] 모델 로드 성공")
except Exception as e:
    print(f"[ERROR] YOLOv5 로드 실패: {e}")
    DETECTOR = None


# =============================
# 얼굴 탐지: YOLO → 실패 시 Haar
# =============================
def _detect_on_frame(detector, frame) -> List[DBox]:
    """
    1단계: YOLOv5로 얼굴 탐지
    2단계: 결과가 없으면 HaarCascade로 한 번 더 탐지
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

    # ---------- 2단계: YOLO 결과가 없으면 Haar ----------
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

    print(f"[DETECT] faces={len(faces)}")
    return faces


# =============================
# 얼굴 매칭 (ArcFace)
# =============================
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


# =============================
# 정책: exclude / force / normal
# =============================
def _should_blur_auto_face(registry: FaceRegistry, b: DBox, auth_exclude_enable: bool) -> bool:
    if not b.enabled:
        # UI에서 끈 박스라면 무조건 모자이크 안 함
        return False

    if auth_exclude_enable and b.identity:
        mode = registry.get_policy(b.identity)

        if mode == "exclude":
            # 등록된 얼굴 중 "exclude" → 항상 제외
            return False

        if mode == "force":
            # 등록된 얼굴 중 "force" → 항상 모자이크
            return True

    # 기본값: 모자이크 대상
    return True


# =============================
# 모자이크 효과
# =============================
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


# =============================
# 이모지 오버레이
# =============================
def _load_emoji_image(emoji_path: Optional[str], emoji_index: int = 0) -> Optional[np.ndarray]:
    if emoji_path:
        path = emoji_path
    else:
        idx = emoji_index if 0 <= emoji_index < len(DEFAULT_EMOJI_FILES) else 0
        path = os.path.join(EMOJI_DIR, DEFAULT_EMOJI_FILES[idx])

    if not os.path.exists(path):
        print(f"[emoji] 파일 없음: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def _overlay_emoji(frame, x, y, w, h, emoji_img):
    if emoji_img is None:
        return

    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    target_w = x2 - x1
    target_h = y2 - y1
    if target_w <= 0 or target_h <= 0:
        return

    emoji_resized = cv2.resize(emoji_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if emoji_resized.shape[2] == 4:  # 알파 채널 존재
        b, g, r, a = cv2.split(emoji_resized)
        emoji_rgb = cv2.merge((b, g, r))
        alpha = (a / 255.0)[..., None]

        roi = frame[y1:y2, x1:x2].astype(float)
        blended = alpha * emoji_rgb + (1 - alpha) * roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = emoji_resized


# =============================
# 메인 처리 함수
# =============================
def process_image(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
    excluded_ids: Optional[List[int]] = None, 
    detect_only: bool = False,
):
    """
    - YOLO가 로드되지 않았더라도(Haar만으로) 최대한 얼굴을 찾아서 처리한다.
    - 얼굴이 아예 안 잡히면 원본 그대로 저장된다.
    """
    if excluded_ids is None:
        excluded_ids = []


    frame = cv2.imread(input_path)
    if frame is None:
        raise IOError(f"이미지 읽기 실패: {input_path}")

    registry = FaceRegistry()

    # 얼굴 자동 탐지 (YOLO + Haar fallback)
    detected_faces = _detect_on_frame(DETECTOR, frame)

    # 얼굴 등록 기반 exclude/force 정책 적용
    if auth_exclude_enable and detected_faces:
        _annotate_authorized_faces(registry, detected_faces, frame)

    # 이모지 로드
    emoji_img = None
    if style == "emoji":
        emoji_img = _load_emoji_image(emoji_path, emoji_index)
        if emoji_img is None:
            style = "blur"  # fallback

    # 각 얼굴 박스에 모자이크/블러/이모지 적용
    for idx,box in enumerate(detected_faces):

        if idx in excluded_ids:
            continue

        if not _should_blur_auto_face(registry, box, auth_exclude_enable):
            continue

        if detect_only:
            # 필요에 따라 색/두께 수정 가능
            cv2.rectangle(
                frame,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                (0, 255, 255),
                2,
            )
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

    # 결과 저장
    ext = os.path.splitext(output_path)[1].lower()
    encode_ext = {
        ".png": ".png",
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".webp": ".webp",
        ".bmp": ".bmp",
    }.get(ext, ".png")

    try:
        ok, enc = cv2.imencode(encode_ext, frame)
        if not ok:
            raise RuntimeError("Encoding failed")
        with open(output_path, "wb") as f:
            f.write(enc)
    except Exception:
        # imencode 실패 시 PIL로 한 번 더 시도
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img.save(output_path)
        except Exception as e:
            raise IOError(f"이미지 저장 실패: {e}")

