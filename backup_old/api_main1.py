from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from flask_cors import CORS
import io, os, cv2, numpy as np, time
from PIL import Image
from deepface import DeepFace
from ai_detect import load_haar_face, detect_faces
from werkzeug.utils import secure_filename
import uuid # For generating unique filenames

# Import studio_image and studio_video for processing
import studio_image
import studio_video


# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "mozik_app", "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "mozik_app", "static")
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
RESULT_FOLDER = os.path.join(STATIC_FOLDER, "results")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
CORS(app)

# ------------------------------
# 기본 설정
# ------------------------------
FACE_DIR = "detected_face"
os.makedirs(FACE_DIR, exist_ok=True)

MODEL_NAME = "VGG-Face"
COSINE_THRESHOLD = 0.40  # 임계값: 낮을수록 엄격
_face_cascade = load_haar_face()
_model_loaded = False
_face_db_cache = None


def ensure_model():
    global _model_loaded
    if not _model_loaded:
        DeepFace.build_model(MODEL_NAME)
        _model_loaded = True


# ------------------------------
# 유틸 함수
# ------------------------------
def cv_imread_from_upload(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def apply_mosaic(image, x, y, w, h, factor=20):
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return image
    small = cv2.resize(roi, (max(1, w//factor), max(1, h//factor)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image


def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ------------------------------
# 얼굴 DB 캐시 로딩
# ------------------------------
def rebuild_face_db():
    """detected_face 폴더의 모든 얼굴 이미지를 Embedding하여 캐시"""
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
            print(f"[WARN] embedding 실패: {fname} ({e})")
    return db


def get_face_db():
    global _face_db_cache
    if _face_db_cache is None:
        _face_db_cache = rebuild_face_db()
    return _face_db_cache


# ------------------------------
# API: 헬스체크
# ------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "faces": len(os.listdir(FACE_DIR))})


# ------------------------------
# API: 얼굴 등록
# ------------------------------
@app.route("/api/face/register", methods=["POST"])
def register_face():
    if "file" not in request.files:
        return jsonify({"error": "file not found"}), 400
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "name missing"}), 400

    f = request.files["file"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    path = os.path.join(FACE_DIR, f"{safe_name}_{ts}.jpg")
    img = cv_imread_from_upload(f)
    cv2.imwrite(path, img)

    # DB 캐시 무효화
    global _face_db_cache
    _face_db_cache = None

    return jsonify({"ok": True, "saved": os.path.basename(path)})


# ------------------------------
# API: 얼굴 목록 조회
# ------------------------------
@app.route("/api/face/list")
def list_faces():
    result = {}
    for fname in sorted(os.listdir(FACE_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        key = fname.split("_")[0]
        result.setdefault(key, []).append(fname)
    return jsonify(result)


# ------------------------------
# API: 얼굴 DB 캐시 새로고침
# ------------------------------
@app.route("/api/face/reload", methods=["POST"])
def reload_faces():
    global _face_db_cache
    _face_db_cache = rebuild_face_db()
    return jsonify({"ok": True, "count": len(_face_db_cache)})


# ------------------------------
# API: 얼굴 감지
# ------------------------------
@app.route("/api/detect/face", methods=["POST"])
def detect_face():
    file = request.files["file"]
    img = cv_imread_from_upload(file)
    faces = detect_faces(img, _face_cascade)
    return jsonify({"faces": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in faces]})


# ------------------------------
# API: 얼굴 모자이크 (등록자는 제외)
# ------------------------------
@app.route("/api/mosaic/face", methods=["POST"])
def mosaic_face():
    factor = int(request.args.get("factor", 20))
    file = request.files["file"]
    img = cv_imread_from_upload(file)
    faces = detect_faces(img, _face_cascade)

    ensure_model()
    db = get_face_db()

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # Embedding 계산
        try:
            rep = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, enforce_detection=False)
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
        except Exception:
            img = apply_mosaic(img, x, y, w, h, factor)
            continue

        # 등록된 얼굴과 비교
        best_name, best_dist = None, 999
        for (name, ref_emb) in db:
            dist = cosine_distance(emb, ref_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        # 매칭 결과 판단
        if best_dist <= COSINE_THRESHOLD:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(img, f"{best_name} ({best_dist:.2f})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            img = apply_mosaic(img, x, y, w, h, factor)

    _, buf = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buf), mimetype="image/png")


# ------------------------------
# API: 얼굴 모자이크 (등록자는 제외)
# ------------------------------
@app.route("/api/mosaic/face", methods=["POST"])
def mosaic_face():
    factor = int(request.args.get("factor", 20))
    file = request.files["file"]
    img = cv_imread_from_upload(file)
    faces = detect_faces(img, _face_cascade)

    ensure_model()
    db = get_face_db()

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # Embedding 계산
        try:
            rep = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, enforce_detection=False)
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
        except Exception:
            img = apply_mosaic(img, x, y, w, h, factor)
            continue

        # 등록된 얼굴과 비교
        best_name, best_dist = None, 999
        for (name, ref_emb) in db:
            dist = cosine_distance(emb, ref_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        # 매칭 결과 판단
        if best_dist <= COSINE_THRESHOLD:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(img, f"{best_name} ({best_dist:.2f})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            img = apply_mosaic(img, x, y, w, h, factor)

    _, buf = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buf), mimetype="image/png")

# --- Page Rendering Routes ---
@app.route("/")
def index():
    """Redirects to the main image processing page."""
    return redirect(url_for("show_image_page"))

@app.route("/image")
def show_image_page():
    """Renders the image upload and processing page."""
    return render_template("image.html")

@app.route("/video")
def show_video_page():
    """Renders the video upload and processing page."""
    return render_template("video.html")


# --- File Upload and Processing Routes (for studio_image/video) ---
@app.route("/upload_image", methods=["POST"])
def upload_image():
    """
    Handles image file upload and processing using studio_image.
    Accepts a multipart/form-data request with 'file' and 'blur_strength'.
    Returns the processed image file directly to the browser.
    """
    if "file" not in request.files:
        return jsonify({"error": "File part is missing"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # --- 1. Save uploaded file securely ---
    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}_{filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    # --- 2. Get processing parameters from the form ---
    try:
        blur_strength = int(request.form.get("blur_strength", 25))
    except (ValueError, TypeError):
        blur_strength = 25
    auth_exclude_enable = request.form.get("auth_exclude", "false").lower() == "true" # The HTML doesn't send this yet, but studio_image expects it. Defaulting to false.

    # --- 3. Process the image using the studio_image module ---
    # Ensure the output has a compatible extension, like .png or .jpg
    _, ext = os.path.splitext(filename)
    output_filename = f"result_{unique_id}{ext}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    try:
        studio_image.process_image(
            input_path=input_path,
            output_path=output_path,
            blur_strength=blur_strength,
            auth_exclude_enable=auth_exclude_enable,
        )
    except Exception as e:
        app.logger.error(f"Image processing failed: {e}")
        return jsonify({"error": f"Failed to process image: {e}"}), 500
    finally:
        # Clean up the original uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)

    # --- 4. Return the processed file directly ---
    return send_file(output_path, mimetype=f"image/{ext.replace('.', '')}", as_attachment=False)


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """
    Handles video file upload and processing using studio_video.
    Accepts a multipart/form-data request with 'file' and 'blur_strength'.
    Returns the processed video file directly to the browser.
    """
    if "file" not in request.files:
        return jsonify({"error": "File part is missing"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # --- 1. Save uploaded file securely ---
    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}_{filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    # --- 2. Get processing parameters from the form ---
    try:
        blur_strength = int(request.form.get("blur_strength", 25))
    except (ValueError, TypeError):
        blur_strength = 25
    auth_exclude_enable = request.form.get("auth_exclude", "false").lower() == "true" # The HTML doesn't send this yet, but studio_video expects it. Defaulting to false.

    # --- 3. Process the video using the studio_video module ---
    # Ensure the output has a compatible extension, like .mp4
    base, _ = os.path.splitext(filename)
    output_filename = f"result_{unique_id}.mp4" # Force .mp4 output as VideoWriter in studio_video uses mp4v
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    try:
        studio_video.process_video(
            input_path=input_path,
            output_path=output_path,
            blur_strength=blur_strength,
            auth_exclude_enable=auth_exclude_enable,
        )
    except Exception as e:
        app.logger.error(f"Video processing failed: {e}")
        return jsonify({"error": f"Failed to process video: {e}"}), 500
    finally:
        # Clean up the original uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)

    # --- 4. Return the processed file directly ---
    return send_file(output_path, mimetype="video/mp4", as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

