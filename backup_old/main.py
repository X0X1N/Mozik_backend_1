from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, uuid, shutil

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "AI API server alive"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1) 파일 저장
    ext = os.path.splitext(file.filename)[1]
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) TODO: 여기서 실제 AI 분석 코드 호출 (모자이크, 얼굴 인식 등)

    # 3) 데모 응답
    result = {
        "success": True,
        "message": "분석 완료 (데모)",
        "saved_path": save_path,
        "detections": [
            {"label": "face", "x": 100, "y": 120, "w": 80, "h": 80}
        ],
    }
    return JSONResponse(content=result)

