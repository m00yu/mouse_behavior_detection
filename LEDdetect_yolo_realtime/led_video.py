"""
input: video
output: water port가 bbox로 탐지
"""
import cv2
import torch
from ultralytics import YOLO

# --------------------- 설정 ---------------------
input_video_path = "C:/Users/NeuRLab/hyunsu/dlcmulti_yoloport_video/videos/short_short_Top.mp4"
output_video_path = "C:/Users/NeuRLab/hyunsu/dlcmulti_yoloport_video/videos/output_video.mp4"
model_path = "C:/Users/NeuRLab/hyunsu/model/yolo_port.pt"

# --------------------- YOLO 모델 로드 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)
print(f"{device}에서 모델 작동 중")

# --------------------- 비디오 로드 ---------------------
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("❌ 비디오 열기 실패")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("✅ 비디오 로드 성공")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측
    results = model.predict(frame, device=device, verbose=False)
    boxes_data = results[0].boxes.data

    # water port 클래스만 필터링
    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
        cls_id = int(cls_id)

        if cls_id in [0, 1, 2, 3] and conf > 0.3:
            lx1, ly1, lx2, ly2 = map(int, (x1, y1, x2, y2))
            label_map = {0: "bottom", 1: "left", 2: "right", 3: "up"}
            label = label_map.get(cls_id, f"port{cls_id}")

            # bbox & label 그리기
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
            cv2.putText(frame, label, (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"✅ 분석 완료! bounding box 포함된 영상 저장됨: {output_video_path}")
