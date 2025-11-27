# app.py  (WSL AI 서버 최종 버전 예시이다.)

import os 
import uvicorn
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware


from api_service import register_face, detect_and_match, reload_registry

from studio_image import process_image
from studio_video import process_video

from face_embedder import get_embedding
from face_registry import FaceRegistry
from ai_detect import load_yolov5, detect_faces_yolov5

# FastAPI 앱 생성 및 공통 설정이다.
# ------------------------------------------------------------
app = FastAPI(title="Mozik AI Server (Final Version)")

# CORS 설정 (KT Cloud 백엔드, 프론트 어디에서든 호출 가능하도록 열어둔다)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 필요하면 도메인으로 좁혀도 된다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)


# ------------------------------------------------------------
# 1. 헬스 체크
# ------------------------------------------------------------
@app.get("/ai_api/health")
async def api_health():
    """
    서버 살아있는지 확인하는 헬스 체크 엔드포인트이다.
    """
    return {"status": "ok", "message": "Mozik Ai Server Running"}


# ------------------------------------------------------------
# 2. 얼굴 등록 API
#    - 사진 + 이름을 받아서, FaceRegistry에 임베딩을 저장한다.
# ------------------------------------------------------------
@app.post("/ai_api/face/register")
async def api_register_face(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    
    img_bytes = await file.read()
    result = register_face(img_bytes, name)  # api_service.register_face 사용
    return JSONResponse(content=result)


# ------------------------------------------------------------
# 3. 얼굴 탐지 + 매칭 API
#    - 사진을 받아서, YOLOv5로 얼굴을 찾고,
#      임베딩을 계산한 후 등록된 얼굴과 매칭 결과를 돌려준다.
# ------------------------------------------------------------
@app.post("/ai_api/detect/face")
async def api_detect_face(
    file: UploadFile = File(...),
):
    """
    얼굴 탐지/매칭용 엔드포인트이다.
    - file: 여러 얼굴이 있을 수 있는 이미지 파일이다.
    반환 값 예시는 대략 다음과 같다.
    {
        "success": true,
        "count": 2,
        "faces": [
            { "x": ..., "y": ..., "w": ..., "h": ...,
              "name": "등록된이름 또는 unknown",
              "distance": 0.42
            },
            ...
        ]
    }
    """
    img_bytes = await file.read()
    result = detect_and_match(img_bytes)  # api_service.detect_and_match 사용
    return JSONResponse(content=result)


# ------------------------------------------------------------
# 4. 레지스트리 리로드 API
#    - face_db 폴더를 다시 읽어서, FaceRegistry 인메모리 캐시를 갱신한다.
#    - 새로운 얼굴을 파일 시스템에 직접 추가한 뒤 갱신하고 싶을 때 사용한다.
# ------------------------------------------------------------
@app.post("/ai_api/face/reload")
async def api_face_reload():
    """
    FaceRegistry를 다시 로드하는 엔드포인트이다.
    """
    result = reload_registry()
return JSONResponse(content=result)


@app.post("/ai_api/mosaic/image")
async def api_mosaic_image(
        file: UploadFile = File(...),
        blur_strength: int = Form(25),
):
    ext = os.path.splitext(file.filename)[1] or ".jpg"

    in_path = f"temp/in_{uuid.uuid4().hex}{ext}"
    out_path = f"temp/out_{uuid.uuid4().hex}{ext}"

    with open(in_path, "wb") as f:
        f.write(await file.read())

    process_image(
            input_path=in_path,
            output_path= out_path,
            blur_strength=int(blur_strength)
    )

    return FileResponse(
            out_path,
            media_type="image/jpeg",
            filename="mosaic.jpg"

    )

@app.post("/ai_api/mosaic/video")
async def api_mosaic_video(
        file: UploadFile = File(...),
        blur_strength: int = From(25),
):
    ext = os.path.splitext(file/filename)[1] or ".mp4"

    in_path = f"temp/in_{uuid.uuid4().hex}{ext}"
    out_path = f"temp/out_{uuid.uuid4().hex}.mp4"

    with open(in_path, "wb") as f:
        f.write(await file.read())

    process_video(
        input_path=in_path,
        output_path=out_path,
        blur_strength=int(blur_strength)
    )

    return FileResponse(
        out_path,
        media_type-"video/mp4",
        filename="mosaic.mp4"
    )

# ------------------------------------------------------------
# uvicorn 실행 진입점이다.
# WSL에서 python app.py 로 실행하면 된다.
# KT Cloud에서는 ssh 역터널로 이 8000 포트를 물려 쓰면 된다.
# ------------------------------------------------------------
if __name__ == "__main__":
    # host, port 는 필요에 따라 변경하면 된다.
    # 역터널 기준으로는 보통 0.0.0.0:8000 을 쓴다.
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

