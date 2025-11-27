import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from core import gaussian_blur_region, expand_box
from detect import load_yolov5, detect_faces_yolov5
from face_registry import FaceRegistry
from face_embedder import get_embedding


# ----------------------------------
# 전역 설정
# ----------------------------------
YOLOV5_WEIGHTS = os.path.join(os.path.dirname(__file__), "best.pt")
DETECT_THUMB_W = 640
FACE_PAD = 0.05  # 얼굴 주변 여유 패딩 비율이다.

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emoji")

_DETECTOR = None  # YOLO 모델 전역 캐시이다.


# ----------------------------------
# 데이터 클래스
# ----------------------------------
@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    identity: Optional[str] = None  # 매칭된 이름(등록 얼굴이면 이름), 없으면 None이다.
    emb: Optional[np.ndarray] = None  # 얼굴 임베딩 캐시이다.


# ----------------------------------
# 내부 유틸 함수들
# ----------------------------------
def _get_detector():
    """YOLOv5 모델을 한 번만 로드해서 재사용하는 함수이다."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = load_yolov5(YOLOV5_WEIGHTS)
    return _DETECTOR


def _detect_on_frame(detector, frame) -> List[FaceBox]:
    """
    단일 프레임에서 얼굴을 탐지해서 FaceBox 리스트로 반환하는 함수이다.
    studio_image.py 의 로직과 최대한 동일하게 맞춘 버전이다.
    """
    h, w = frame.shape[:2]
    faces: List[FaceBox] = []

    # YOLO 모델이 없으면 바로 리턴
    if detector is None:
        print("[VIDEO_DETECT] detector is None", flush=True)
        return faces

    # ------------ 1단계: 리사이즈 (너무 큰 영상일 때) ------------
    if w > DETECT_THUMB_W:
        scale = DETECT_THUMB_W / float(w)
        small = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale)))
        )
    else:
        scale = 1.0
        small = frame

    # ------------ 2단계: YOLO로 얼굴 박스 얻기 ------------
    try:
        # conf_thres 기본값(0.4)을 사용한다.
        rects_small = detect_faces_yolov5(detector, small)
    except Exception as e:
        print(f"[VIDEO_DETECT] YOLO detect 실패: {e}", flush=True)
        rects_small = []

    pad = float(FACE_PAD)

    # ------------ 3단계: 원본 해상도로 좌표 되돌리기 + 패딩 적용 ------------
    for (x, y, w0, h0) in rects_small:
        # 리사이즈했으면 원래 좌표로 복구
        if scale != 1.0:
            x, y, w0, h0 = int(x / scale), int(y / scale), int(w0 / scale), int(h0 / scale)

        # core.expand_box 를 이용해 약간 확장된 박스 계산
        ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)

        # 너무 작은 박스는 무시
        if ew >= 12 and eh >= 12:
            faces.append(FaceBox(ex, ey, ew, eh))

    print(f"[VIDEO_DETECT] faces={len(faces)}", flush=True)
    return faces

  

def _crop_face_with_pad(frame, box: FaceBox) -> Optional[np.ndarray]:
    """
    FACE_PAD 비율만큼 확장된 영역으로 얼굴을 잘라내서 반환하는 함수이다.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = expand_box(box.x, box.y, box.w, box.h, FACE_PAD)

    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    x2 = max(0, min(w, x + bw))
    y2 = max(0, min(h, y + bh))

    bw = x2 - x
    bh = y2 - y
    if bw <= 0 or bh <= 0:
        return None

    return frame[y:y2, x:x2]


def _annotate_authorized_faces(registry: FaceRegistry, faces: List[FaceBox], frame):
    """
    FaceRegistry를 이용해 각 FaceBox에 identity 를 채워 넣는 함수이다.
    auth_exclude_enable 이 True일 때만 호출하면 된다.
    """
    for box in faces:
        crop = _crop_face_with_pad(frame, box)
        if crop is None:
            continue

        try:
            emb = get_embedding(crop)
        except Exception:
            emb = None

        if emb is None:
            continue

        box.emb = emb

        try:
            name, dist = registry.match(emb)
        except Exception:
            name, dist = None, None

        box.identity = name


def _should_blur_auto_face(
    registry: FaceRegistry,
    box: FaceBox,
    auth_exclude_enable: bool,
) -> bool:
    """
    어떤 얼굴을 블러/모자이크 할지 결정하는 정책 함수이다.

    - auth_exclude_enable == False 이면: 모든 얼굴을 블러 처리한다.
    - auth_exclude_enable == True 이면:
        box.identity 가 존재하면 "등록된 얼굴" 이라고 보고 모자이크에서 제외한다.
    """
    if not auth_exclude_enable:
        return True

    # 간단한 정책: 등록된 얼굴(identity가 있는 경우)은 제외한다.
    if box.identity:
        return False

    return True


def _load_emoji_image(emoji_path: Optional[str], emoji_index: int) -> Optional[np.ndarray]:
    """
    이모지 이미지를 읽어서 반환하는 함수이다.
    - emoji_path 가 주어지면 그 경로를 최우선으로 사용한다.
    - 아니면 EMOJI_DIR/emoji_{index}.png 형식으로 찾는다.
    """
    candidate_paths = []

    if emoji_path:
        candidate_paths.append(emoji_path)

    # 기본 경로: emoji_0.png, emoji_1.png 등의 이름을 가정한다.
    filename = f"emoji_{emoji_index}.png"
    candidate_paths.append(os.path.join(EMOJI_DIR, filename))

    for path in candidate_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

    return None


def _overlay_emoji(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    emoji_img: Optional[np.ndarray],
):
    """얼굴 영역 위에 이모지 이미지를 덮어쓰는 함수이다."""
    if emoji_img is None:
        return

    fh, fw = frame.shape[:2]

    x = max(0, min(fw - 1, x))
    y = max(0, min(fh - 1, y))
    x2 = max(0, min(fw, x + w))
    y2 = max(0, min(fh, y + h))

    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        return

    # 이모지를 얼굴 크기에 맞게 리사이즈
    emoji_resized = cv2.resize(emoji_img, (w, h), interpolation=cv2.INTER_AREA)

    if emoji_resized.shape[2] == 4:
        # RGBA (알파 포함) 이면 알파 블렌딩
        emoji_rgb = emoji_resized[:, :, :3]
        alpha = emoji_resized[:, :, 3:] / 255.0

        roi = frame[y:y2, x:x2]
        if roi.shape[:2] != emoji_rgb.shape[:2]:
            # 혹시 크기 안 맞으면 한 번 더 맞춰준다.
            emoji_rgb = cv2.resize(emoji_rgb, (roi.shape[1], roi.shape[0]))
            alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))

        frame[y:y2, x:x2] = (alpha * emoji_rgb + (1 - alpha) * roi).astype(
            np.uint8
        )
    else:
        # 알파가 없으면 그냥 덮어쓰기
        frame[y:y2, x:x2] = emoji_resized[:, :, :3]


def _apply_mosaic(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    mosaic_size: int = 15,
):
    """얼굴 영역에 픽셀 모자이크를 적용하는 함수이다."""
    fh, fw = frame.shape[:2]

    x = max(0, min(fw - 1, x))
    y = max(0, min(fh - 1, y))
    x2 = max(0, min(fw, x + w))
    y2 = max(0, min(fh, y + h))

    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        return

    roi = frame[y:y2, x:x2]
    # 너무 작지 않게 최소 1x1 보장
    small_w = max(1, w // mosaic_size)
    small_h = max(1, h // mosaic_size)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y2, x:x2] = mosaic

# ----------------------------------
# 메인 엔트리: process_video
# ----------------------------------
def process_video(
    input_path: str,
    output_path: str,
    blur_strength: int,
    auth_exclude_enable: bool = False,
    style: str = "blur",
    emoji_path: Optional[str] = None,
    emoji_index: int = 0,
):
    """
    업로드된 영상 파일을 프레임 단위로 읽어,
    - 얼굴이 잡히면: 얼굴 영역만 blur/mosaic/emoji 처리
    - 얼굴이 하나도 안 잡히면: 프레임 전체를 blur 처리한 뒤
      output_path 에 mp4 로 저장하는 함수이다.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 25.0  # FPS 정보가 깨져 있으면 기본값

    # 출력 비디오 준비
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 얼굴 등록 레지스트리 (등록 얼굴 제외 기능용)
    registry = FaceRegistry()

    # 이모지 스타일이면 이모지 이미지를 미리 로드
    emoji_img = None
    if style == "emoji":
        emoji_img = _load_emoji_image(emoji_path, emoji_index)
        if emoji_img is None:
            # 이모지 로딩 실패 시, 안전하게 blur 스타일로 대체
            style = "blur"

    detector = _get_detector()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ------------------------------
            # 1) 얼굴 탐지
            # ------------------------------
            faces = _detect_on_frame(detector, frame)

            # ------------------------------
            # 2) 얼굴이 하나도 안 잡힌 경우:
            #    프레임 전체를 블러 처리
            # ------------------------------
            if not faces:
                gaussian_blur_region(
                    frame,
                    0,
                    0,
                    width,
                    height,
                    blur_strength,
                )
                writer.write(frame)
                continue

            # ------------------------------
            # 3) 등록 얼굴/정책 반영
            # ------------------------------
            if auth_exclude_enable:
                _annotate_authorized_faces(registry, faces, frame)

            # ------------------------------
            # 4) 스타일별 처리
            # ------------------------------
            for box in faces:
                # 등록 얼굴 등으로 모자이크 제외 대상이면 건너뛰기
                if not _should_blur_auto_face(registry, box, auth_exclude_enable):
                    continue

                if style == "blur":
                    gaussian_blur_region(
                        frame,
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        blur_strength,
                    )

                elif style == "mosaic":
                    _apply_mosaic(
                        frame,
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        mosaic_size=max(8, min(30, int(blur_strength))),
                    )

                elif style == "emoji":
                    _overlay_emoji(frame, box.x, box.y, box.w, box.h, emoji_img)

                else:
                    # 알 수 없는 스타일이면 기본적으로 blur
                    gaussian_blur_region(
                        frame,
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        blur_strength,
                    )

            # 한 프레임 최종 결과 쓰기
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

