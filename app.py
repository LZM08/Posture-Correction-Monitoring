from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import time
from winotify import Notification
import threading
import json
import os
import torch  # PyTorch 임포트 추가

app = Flask(__name__)

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # GPU가 사용되는지 확인

# YOLO 모델 로드 (GPU로 로드)
model = YOLO('yolo11x-pose.pt').to(device)  # .to(device)를 추가하여 GPU로 로드

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 알림 관련 변수
last_notification_time = 0
notification_interval = 10
current_posture_status = {"status": "인식 중...", "angle": 0.0, "confidence": 0.0}

# 아이콘 경로 설정
icon_path = os.path.abspath("warning-sign.ico")  # 실제 아이콘 파일 경로로 변경 필요

def notify_bad_posture(message):
    """Windows 알림 표시"""
    toast = Notification(
        app_id="Posture Detection",
        title="자세 교정 알림",
        msg=message,
        icon=icon_path,
        duration="short"
    )
    
    # 스트레칭 영상 버튼 추가
    toast.add_actions(
        label="의자에서 올바르게 앉는법",
        launch="https://youtu.be/3tCjbwNu9l0?si=VihaD6APdPufVMQn"
    )
    
    # 알림 표시
    toast.show()

def calculate_head_angle(nose, ear):
    """머리 기울기 각도 계산"""
    if nose is None or ear is None:
        return 0.0
    
    dx = ear[0] - nose[0]
    dy = ear[1] - nose[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return abs(angle)

def generate_frames():
    global last_notification_time, current_posture_status
    
    try:
        font = ImageFont.truetype("malgun.ttf", 30)
    except:
        font = ImageFont.truetype("arial.ttf", 30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)  # GPU에서 처리하도록 자동으로 변경됩니다
        frame_info = {
            "head_angle": 0.0,
            "confidence": 0.0,
            "posture_status": "인식 중...",
            "color": (255, 255, 255)
        }
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints[0]
            
            # keypoints가 정상적으로 감지되었는지 확인
            if keypoints.conf is not None and keypoints.xy is not None and len(keypoints.conf) > 0 and len(keypoints.xy) > 0:
                keypoints_xy = keypoints.xy[0].cpu().numpy()
                keypoints_conf = keypoints.conf[0].cpu().numpy()
                
                nose = keypoints_xy[0] if keypoints_conf[0] > 0.5 else None
                left_ear = keypoints_xy[3] if keypoints_conf[3] > 0.5 else None
                right_ear = keypoints_xy[4] if keypoints_conf[4] > 0.5 else None
                
                if nose is not None and (left_ear is not None or right_ear is not None):
                    head_y = float(nose[1])
                    frame_height = float(frame.shape[0])
                    
                    ear = left_ear if left_ear is not None else right_ear
                    head_angle = calculate_head_angle(nose, ear)
                    
                    frame_info["head_angle"] = float(round(head_angle, 1))
                    frame_info["confidence"] = float(round(float(keypoints_conf[0]) * 100, 1))
                    
                    if head_y > frame_height * 0.4 or head_angle > 45:
                        frame_info["posture_status"] = "자세 나쁨"
                        frame_info["color"] = (0, 0, 255)
                        
                        current_time = time.time()
                        if current_time - last_notification_time >= notification_interval:
                            threading.Thread(
                                target=notify_bad_posture,
                                args=("자세를 올바르게 바꾸세요.",)
                            ).start()
                            last_notification_time = current_time
                    else:
                        frame_info["posture_status"] = "자세 좋음"
                        frame_info["color"] = (0, 255, 0)
                else:
                    frame_info["posture_status"] = "키포인트 신뢰도 낮음"
                    frame_info["color"] = (255, 0, 0)  # 신뢰도 낮음, 빨간색 표시
                
                current_posture_status["status"] = frame_info["posture_status"]
                current_posture_status["angle"] = float(frame_info["head_angle"])
                current_posture_status["confidence"] = float(frame_info["confidence"])
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)
                
                status_text = f"{frame_info['posture_status']} | 각도: {frame_info['head_angle']}° | 정확도: {frame_info['confidence']}%"
                draw.text((20, 20), status_text, font=font, 
                         fill=(frame_info["color"][0], frame_info["color"][1], frame_info["color"][2]))
                
                # YOLO 모델의 plot 결과를 사용
                frame = results[0].plot()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_status():
    while True:
        data = json.dumps({
            "status": current_posture_status["status"],
            "angle": current_posture_status["angle"],
            "confidence": current_posture_status["confidence"]
        })
        yield f"data: {data}\n\n"
        time.sleep(0.1)

@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_stream')
def status_stream():
    return Response(generate_status(), 
                   mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
