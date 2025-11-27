import os
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

from face_embedder import get_embedding
from face_registry import FaceRegistry
from ai_detect import load_yolov5, detect_faces_yolov5

YOLO_MODEL = None
REGISTRY: Optional[FaceRegistry] = None


def _get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = load_yolov5("best.pt")
    return YOLO_MODEL


def _get_registry() -> FaceRegistry:
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = FaceRegistry(db_path="face_db", sim_threshold=0.40)
    return REGISTRY


def bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img


def register_face(img_bytes: bytes, name: str) -> Dict[str, Any]:
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
        "box": {"x": x, "y": y, "w": bw, "h": bh},
    }


def detect_and_match(img_bytes: bytes) -> Dict[str, Any]:
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
                "error": f"embedding_failed: {e}",
            })
            continue

        name, dist = reg.match(emb)
        results.append({
            "x": x, "y": y, "w": bw, "h": bh,
            "name": name,
            "distance": float(dist),
        })

    return {
        "success": True,
        "count": len(results),
        "faces": results,
    }


def reload_registry() -> Dict[str, Any]:
    global REGISTRY
    REGISTRY = FaceRegistry(db_path="face_db", sim_threshold=0.40)
    return {"success": True}
