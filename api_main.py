#############################################
# ai_api/api_main.py  (AI 서버)
#############################################

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import io
import os
import cv2
import numpy as np
import time
import tempfile
import uuid
import traceback

from PIL import Image

# YOLO + DeepFace
from deepface import DeepFace
from detect import load_yolov5, detect_faces_yolov5

from api_service import register_face, detect_and_match, reload_registry
from studio_image import process_image
from studio_video import process_video

# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FACE_DIR = os.path.join(BASE_DIR, "detected_face")
os.makedirs(FACE_DIR, exist_ok=True)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_NAME = "VGG-Face"
COSINE_THRESHOLD = 0.40

YOLO_WEIGHT = os.path.join(BASE_DIR, "best.pt")
detector = load_yolov5(YOLO_WEIGHT)

app = Flask(__name__)
CORS(app)

_model_loaded = False
_face_db_cache = None


#############################################
# 유틸 함수
#############################################

def safe_log(prefix, e):
    print(f"[{prefix}] {str(e)}")
    print(traceback.format_exc())


def safe_read_image(upload_file):
    """Flask FileStorage → OpenCV BGR 이미지로 안전하게 변환하는 함수이다."""
    try:
        img = Image.open(upload_file.stream).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"이미지 디코딩 실패: {e}")


def safe_embedding(face_roi):
    """DeepFace 임베딩을 구하되, 실패하면 None을 리턴하는 함수이다."""
    try:
        rep = DeepFace.represent(face_roi, model_name=MODEL_NAME, enforce_detection=False)
        return rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
    except Exception as e:
        safe_log("EMBED", e)
        return None


def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def ensure_model():
    """DeepFace 모델을 한 번만 로드하는 함수이다."""
    global _model_loaded
    if not _model_loaded:
        DeepFace.build_model(MODEL_NAME)
        _model_loaded = True


def apply_mosaic(img, x, y, w, h, factor=20):
    """
    얼굴 영역에 픽셀 모자이크를 적용하는 유틸 함수이다.
    /api/mosaic/face 에서 사용한다.
    """
    h_img, w_img = img.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    small_w = max(1, (x2 - x1) // factor)
    small_h = max(1, (y2 - y1) // factor)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    img[y1:y2, x1:x2] = mosaic
    return img


#############################################
# 얼굴 DB 로딩
#############################################

def rebuild_face_db():
    ensure_model()
    db = []
    for fname in os.listdir(FACE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(FACE_DIR, fname)
        try:
            rep = DeepFace.represent(img_path=path, model_name=MODEL_NAME, enforce_detection=False)
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
            name = fname.split("_")[0]
            db.append((name, np.array(emb)))
        except Exception as e:
            safe_log("DB_EMBED", e)
    return db


def get_face_db():
    global _face_db_cache
    if _face_db_cache is None:
        _face_db_cache = rebuild_face_db()
    return _face_db_cache


#############################################
# 헬스 체크
#############################################

@app.route("/api/health")
def health():
    try:
        return jsonify({"status": "ok", "faces": len(os.listdir(FACE_DIR))})
    except Exception as e:
        safe_log("HEALTH", e)
        return jsonify({"error": "health check failed"}), 500


#############################################
# 얼굴 검출 (YOLOv5)
#############################################

@app.route("/api/detect/face", methods=["POST"])
def api_detect_face():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file not found"}), 400

        img = safe_read_image(request.files["file"])
        faces = detect_faces_yolov5(detector, img)
        return jsonify({"faces": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in faces]})
    except Exception as e:
        safe_log("DETECT", e)
        return jsonify({"error": "detect failed", "details": str(e)}), 500


#############################################
# 얼굴 모자이크 (DeepFace + YOLO)
#############################################

@app.route("/api/mosaic/face", methods=["POST"])
def api_mosaic_face():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file not found"}), 400

        factor = int(request.args.get("factor", 20))
        img = safe_read_image(request.files["file"])

        ensure_model()
        db = get_face_db()

        faces = detect_faces_yolov5(detector, img)

        for (x, y, w, h) in faces:
            roi = img[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            emb = safe_embedding(roi)
            if emb is None:
                img = apply_mosaic(img, x, y, w, h, factor)
                continue

            best_name, best_dist = None, 999
            for name, ref_emb in db:
                dist = cosine_distance(emb, ref_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            if best_dist <= COSINE_THRESHOLD:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)
                cv2.putText(
                    img,
                    f"{best_name} ({best_dist:.2f})",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                img = apply_mosaic(img, x, y, w, h, factor)

        _, buf = cv2.imencode(".png", img)
        return send_file(io.BytesIO(buf), mimetype="image/png")

    except Exception as e:
        safe_log("MOSAIC", e)
        return jsonify({"error": "mosaic failed", "details": str(e)}), 500


#############################################
# 이미지 모자이크 (studio_image 사용)
#############################################

@app.route("/ai_api/mosaic/image", methods=["POST"])
def ai_mosaic_image():
    """
    KT Cloud 웹 서버에서 호출하는 이미지 모자이크 엔드포인트이다.

    - request.files["file"]        : 원본 이미지
    - blur_strength                : 블러 강도 (기본 25)
    - style                        : "blur" / "mosaic" / "emoji"
    - auth_exclude_enable          : 등록 얼굴 정책 사용 여부
    - detect_only                  : 탐지만 할지 여부 (확장용)
    - excluded_ids / excluded_ids[]: 모자이크에서 제외할 얼굴 index 리스트 (확장용)
    """
    try:
        # 1) 파일 확인
        if "file" not in request.files:
            return "file not found", 400

        up_file = request.files["file"]
        if up_file.filename == "":
            return "empty filename", 400

        filename = secure_filename(up_file.filename)

        # 2) 저장 경로
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        output_path = os.path.join(RESULT_DIR, f"result_{uuid.uuid4().hex}.png")

        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        up_file.save(input_path)

        # 3) 옵션 읽기 ----------------------------

        # 블러 강도
        blur_raw = request.form.get("blur_strength", request.args.get("blur_strength", "25"))
        try:
            blur_strength = int(blur_raw)
        except (TypeError, ValueError):
            blur_strength = 25

        # 스타일
        style = request.form.get("style", request.args.get("style", "blur")) or "blur"

        # 등록 얼굴 정책 사용 여부
        auth_raw = request.form.get(
            "auth_exclude_enable",
            request.args.get("auth_exclude_enable", "0"),
        )
        auth_exclude_enable = str(auth_raw).lower() in ("1", "true", "yes", "on")

        # detect_only
        detect_only_raw = request.form.get(
            "detect_only",
            request.args.get("detect_only", "0"),
        )
        detect_only = str(detect_only_raw).lower() in ("1", "true", "yes", "on")

        # excluded_ids: "0,2,5" 또는 excluded_ids[] 형태 모두 지원
        excluded_ids: list[int] = []

        raw_list = request.form.getlist("excluded_ids[]")
        raw_single = request.form.get("excluded_ids", "")

        if raw_list:
            for x in raw_list:
                x = x.strip()
                if x.isdigit():
                    excluded_ids.append(int(x))
        elif raw_single:
            for token in raw_single.split(","):
                token = token.strip()
                if token.isdigit():
                    excluded_ids.append(int(token))

        # 이모지 옵션 (style == "emoji" 일 때만 의미 있음)
        emoji_path = request.form.get("emoji_path", request.args.get("emoji_path"))
        emoji_index_raw = request.form.get("emoji_index", request.args.get("emoji_index", "0"))
        try:
            emoji_index = int(emoji_index_raw)
        except (TypeError, ValueError):
            emoji_index = 0

        # 4) 실제 모자이크 처리 --------------------
        # ⚠️ 주의: studio_image.process_image 시그니처가
        #   (input_path, output_path, blur_strength, auth_exclude_enable,
        #    style, emoji_path, emoji_index, detect_only, excluded_ids)
        # 를 받도록 수정되어 있어야 한다.
        process_image(
            input_path=input_path,
            output_path=output_path,
            blur_strength=blur_strength,
            auth_exclude_enable=auth_exclude_enable,
            style=style,
            emoji_path=emoji_path,
            emoji_index=emoji_index,

            
        )

        # 5) 결과 이미지 반환
        return send_file(output_path, mimetype="image/png")

    except Exception as e:
        safe_log("AI_MOSAIC_IMAGE", e)
        return jsonify({"error": "AI image mosaic failed", "detail": str(e)}), 500


#############################################
# 영상 모자이크 (studio_video 사용)
#############################################

@app.route("/ai_api/mosaic/video", methods=["POST"])
def ai_mosaic_video():
    """
    업로드된 영상을 studio_video.process_video로 모자이크 처리해서 반환하는 엔드포인트이다.
    """
    try:
        # 1) 파일 유효성 검사
        if "file" not in request.files:
            return "file not found", 400

        up_file = request.files["file"]
        if up_file.filename == "":
            return "empty filename", 400

        filename = secure_filename(up_file.filename)

        # 2) 저장 경로
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        output_path = os.path.join(RESULT_DIR, f"result_{uuid.uuid4().hex}.mp4")

        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        up_file.save(input_path)

        # 3) 옵션 읽기
        blur_raw = request.form.get("blur_strength", request.args.get("blur_strength", "25"))
        try:
            blur_strength = int(blur_raw)
        except (TypeError, ValueError):
            blur_strength = 25

        style = request.form.get("style", request.args.get("style", "blur")) or "blur"

        auth_raw = request.form.get(
            "auth_exclude_enable",
            request.args.get("auth_exclude_enable", "0"),
        )
        auth_exclude_enable = str(auth_raw).lower() in ("1", "true", "yes", "on")

        emoji_index_raw = request.form.get("emoji_index", request.args.get("emoji_index", "0"))
        try:
            emoji_index = int(emoji_index_raw)
        except (TypeError, ValueError):
            emoji_index = 0

        emoji_path = None  # 필요하면 실제 경로를 넣으면 됨이다.

        # 4) 실제 모자이크 처리
        process_video(
            input_path=input_path,
            output_path=output_path,
            blur_strength=blur_strength,
            auth_exclude_enable=auth_exclude_enable,
            style=style,
            emoji_path=emoji_path,
            emoji_index=emoji_index,
        )

        # 5) 처리된 영상 반환
        return send_file(output_path, mimetype="video/mp4")

    except Exception as e:
        safe_log("AI_MOSAIC_VIDEO", e)
        return jsonify({"error": "AI video mosaic failed", "details": str(e)}), 500


#############################################
# 에러 핸들러
#############################################

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found"}), 404


@app.errorhandler(Exception)
def global_exception(e):
    safe_log("GLOBAL", e)
    return jsonify({"error": "Internal Error", "details": str(e)}), 500


#############################################
# 서버 시작
#############################################

if __name__ == "__main__":
    print("[AI_API] 서버 시작합니다.")
    app.run(host="0.0.0.0", port=8000, debug=False)

