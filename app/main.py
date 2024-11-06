from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import subprocess
import torch
from detect import run_prediction
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
	"*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 모델을 선언합니다.
model_eye = None
model_nose = None
model_mouth = None
final_layer = None

@app.get("/model")
def read_root():
    return {"Hello": "World"}

@app.get("/model/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.get("/model/trigger")
def read_trigger():
    try:
        result = subprocess.run(
            ['/bin/bash', '/home/ubuntu/model/run_model.sh'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}

@app.get("/model/trigger1")
def read_trigger1():
    try:
        result = subprocess.run(
            ['/bin/bash', '/home/ubuntu/model/run_model1.sh'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}

class AnalysisResult(BaseModel):
    result: float


@app.post("/model/photo/detection", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    # 업로드된 이미지를 임시 파일로 저장
    temp_image_path = "/tmp/temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(await file.read())

    # 가중치 파일 경로 지정
    weights_path = "/home/ubuntu/detector/deepfake-detector/models/best.pth"

    # GPU ID (필요한 경우 설정)
    gpu_id = 5

    try:
        # detect.py의 run_prediction 함수 호출
        fake_prob = run_prediction(temp_image_path, weights_path, gpu_id)
        return {"result": fake_prob}

    except RuntimeError as e:
        # 예측 오류 시 에러 메시지 반환
        return {"status": "error", "error": str(e)}

    finally:
        # 임시 파일 삭제
        os.remove(temp_image_path)


@app.post("/model/photo/protect")
async def generate_perturbed_image():
    # `inference.py` 파일 경로와 작업 디렉토리 설정
    script_path = "/home/ubuntu/attack/DeepFake_Disrupter/inference.py"
    working_directory = "/home/ubuntu/attack/DeepFake_Disrupter"

    # `inference.py` 실행
    try:
        result = subprocess.run(
            ["python3", script_path],
            check=True,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Script output:", result.stdout)

    except subprocess.CalledProcessError as e:
        # 오류 발생 시 HTTP 예외 발생
        raise HTTPException(status_code=500, detail=f"Script error: {e.stderr}")

    # 결과 이미지 경로
    output_image_path = os.path.join(working_directory, "output_image.jpg")

    # 결과 이미지 파일이 생성되었는지 확인
    if not os.path.exists(output_image_path):
        raise HTTPException(status_code=500, detail="Output image not found")

    # 결과 이미지 반환
    return FileResponse(output_image_path, media_type="image/jpeg", filename="output_image.jpg")