"""
input: 실시간 웹캠
output: water port 탐지 되면 DAQ TTL pulse

실행: python led_realtime_inference.py
"""
import cv2
import torch
import time
import nidaqmx
from ultralytics import YOLO

# --------------------- 설정 ---------------------
model_path = "C:/Users/NeuRLab/hyunsu/LEDdetect_yolo_realtime/runs/detect/train/weights/best.pt"

DAQ_CHANNEL = "Dev2/ao0"
TTL_VOLTAGE = 5.0 # output signal (V)
TTL_DURATION = 1  # 초 단위

# --------------------- YOLO 모델 로드 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)
print(f"{device}에서 모델 돌아가는중")

# --------------------- DAQ 설정 ---------------------
daq_task = nidaqmx.Task()
daq_task.ao_channels.add_ao_voltage_chan(DAQ_CHANNEL, min_val=-5.0, max_val=5.0)

# --------------------- 웹캠 열기 ---------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠이 정상적으로 열렸습니다. 종료하려면 'q'를 누르세요.")

# TTL 중복 방지를 위한 타이밍 제어
last_ttl_time = 0
ttl_cooldown = 1.0  # 최소 TTL 간격 (초)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # # YOLO 예측 수행
    # cv2.imwrite(f"webcam_debug_frame_{int(time.time())}.jpg", frame)
    results = model.predict(frame, device=device, verbose=False, imgsz=640)
    boxes_data = results[0].boxes.data
    # print(f"📷 Frame 분석됨. 감지된 개수: {len(boxes_data)}")

    best_box = None
    best_conf = 0.0
    best_cls_id = None

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
        cls_id = int(cls_id)

        if cls_id in [0, 1, 2, 3] and conf > best_conf:
            best_box = (x1, y1, x2, y2)
            best_conf = conf
            best_cls_id = cls_id

    # water port 감지되면 TTL 출력
    if best_box is not None:
        now = time.time()
        if now - last_ttl_time > ttl_cooldown:
            print(f"🔔 water port 감지됨 (Class {best_cls_id}) → TTL 신호 출력!")
            daq_task.write(TTL_VOLTAGE)
            time.sleep(TTL_DURATION)
            daq_task.write(0.0)
            last_ttl_time = now

        # 시각화: bbox + 클래스 이름 표시
        lx1, ly1, lx2, ly2 = map(int, best_box)
        label_map = {0: "bottom", 1: "left", 2: "right", 3: "up"}
        label = label_map.get(best_cls_id, f"port{best_cls_id}")
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
        cv2.putText(frame, label, (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 프레임 보여주기
    cv2.imshow("YOLO Water Port Detection (Webcam)", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------- 종료 정리 ---------------------
cap.release()
cv2.destroyAllWindows()
daq_task.close()
print("🛑 웹캠 및 DAQ 세션 종료.")
