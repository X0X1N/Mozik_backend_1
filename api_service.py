# api_service.py
import os
import io
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

from face_embedder import get_embedding
from face_registry import FaceRegistry
from detect import load_yolov5, detect_faces_yolov5

YOLO_MODEL = None
REGISTRY: Optional[FaceRegistry] = None

def _get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = load_yolov5("best.pt")
    return YOLO_MODEL

def _get_registry() -> FaceRegistry:
    """
    얼굴 임베딩 DB를 관리하는 레지스트리를 전역 1개만 사용한다.
    """
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = FaceRegistry(db_path="face_db", sim_threshold=0.40)
    return REGISTRY

def bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    """
    업로드된 이미지 바이트를 OpenCV BGR 이미지로 변환한다.
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img

def bgr_to_png_bytes(img: np.ndarray) -> bytes:
    """
    BGR 이미지를 PNG 바이트로 변환한다.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("이미지 인코딩 실패")
    return buf.tobytes()


def register_face(img_bytes: bytes, name: str) -> Dict[str, Any]:
    """
    업로드된 사진에서 가장 큰 얼굴 1개를 찾아서 해당 이름으로 DB에 등록한다.
    """
    img = bytes_to_bgr(img_bytes)
    h, w = img.shape[:2]


    model = _get_yolo_model()
    boxes: List[Tuple[int, int, int, int]] = detect_faces_yolov5(img, model=model)

    if not boxes:
        return {"success": False, "reason": "얼굴을 찾지 못함"}

    def area(b):
        _, _, bw, bh = b
        return bw * bh

    x, y, bw, bh = max(boxes, key=area)


    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    face_crop = img[y0:y1, x0:x1]


    emb = get_embedding(face_crop)

    reg = _get_registry()
    reg.add(name, emb)

    return {
        "success": True,
        "name": name,
        "box": {"x": x, "y": y, "w": bw, "h": bh}
    }


def detect_and_match(img_bytes: bytes) -> Dict[str, Any]:
    """
    업로드된 이미지에서 얼굴을 찾고,
    각 얼굴이 등록된 사람인지 매칭한 결과를 JSON 형태로 반환한다.
    """
    img = bytes_to_bgr(img_bytes)
    h, w = img.shape[:2]

    model = _get_yolo_model()
    boxes: List[Tuple[int, int, int, int]] = detect_faces_yolov5(img, model=model)

    reg = _get_registry()

    results = []
    for (x, y, bw, bh) in boxes:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        face_crop = img[y0:y1, x0:x1]

        try:
            emb = get_embedding(face_crop)
        except Exception as e:

            results.append({
                "x": x, "y": y, "w": bw, "h": bh,
                "name": None,
                "distance": None,
                "error": f"embedding_failed: {e}"
            })
            continue

        name, dist = reg.match(emb)
        results.append({
            "x": x, "y": y, "w": bw, "h": bh,
            "name": name,
            "distance": float(dist)
        })

    return {
        "success": True,
        "count": len(results),
        "faces": results
    }

# -----------------------
# 3) DB 리로드(정책/파일 다시 읽기)
# -----------------------
def reload_registry() -> Dict[str, Any]:
    """
    face_db 폴더 내용을 다시 읽어서 레지스트리를 새로 만든다.
    """
    global REGISTRY
    REGISTRY = FaceRegistry(db_path="face_db", sim_threshold=0.40)
    return {"success": True}

