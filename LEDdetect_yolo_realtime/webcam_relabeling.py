"""
웹캠 각도나 LED 조명 등이 달라지면 YOLO가 제대로 탐지 못함
따라서 새로 데이터 라벨링 해서 학습시켜야 함 (roboflow 이용)
"""
"""
웹캠 프레임 샘플링하여 이미지 저장
출력: ./samples/frame_0001.jpg, ...
"""

import cv2
import os

# --------------------- 설정 ---------------------
output_dir = "./samples"
os.makedirs(output_dir, exist_ok=True)

frame_interval = 30   # 몇 프레임마다 저장할지 (30이면 약 1초 간격 @30fps)
max_frames = 300      # 최대 저장할 프레임 수
cam_index = 0         # 기본 웹캠

# --------------------- 웹캠 열기 ---------------------
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠에서 이미지 샘플링 중... (종료하려면 'q')")

saved = 0
frame_idx = 0

while saved < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    frame_idx += 1

    # 'q' 누르면 수동 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print(f"✅ 완료! 총 {saved}개의 이미지 저장됨 → {output_dir}")
