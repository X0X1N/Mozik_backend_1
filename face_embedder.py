import numpy as np
from deepface import DeepFace

_EMBED_MODEL = None
def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = DeepFace.build_model("Facenet512")  # 또는 "VGG-Face"
    return _EMBED_MODEL

def get_embedding(bgr_face_np: np.ndarray) -> np.ndarray:
    import cv2
    rgb = cv2.cvtColor(bgr_face_np, cv2.COLOR_BGR2RGB)
    reps = DeepFace.represent(rgb, model_name="Facenet512", enforce_detection=False, detector_backend="skip", model=_get_model())
    vec = np.array(reps[0]["embedding"], dtype=np.float32)
    return vec
