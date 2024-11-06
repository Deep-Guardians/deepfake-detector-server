import subprocess

def run_prediction(image_path, weights_path, gpu_id=0):
    """
    predict_multimodal.py 스크립트를 실행하여 딥페이크 확률을 반환합니다.
    """
    script_path = "/home/ubuntu/detector/deepfake-detector/predict_multimodal.py"
    working_directory = "/home/ubuntu/detector/deepfake-detector"

    # Python 스크립트 실행
    try:
        result = subprocess.run(
            ["python3", script_path, "--image_path", image_path, "--weights_path", weights_path, "--gpu", str(gpu_id)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_directory  # 작업 디렉토리 설정
        )

        # predict_multimodal.py의 출력에서 fake_prob 값 추출
        output_lines = result.stdout.strip().splitlines()
        for line in output_lines:
            if "딥페이크 확률" in line:
                fake_prob_str = line.split("딥페이크 확률: ")[1].split("%")[0].strip()
                fake_prob = float(fake_prob_str)  # 딥페이크 확률 값을 float로 변환
                return fake_prob

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Prediction script error: {e.stderr}")


