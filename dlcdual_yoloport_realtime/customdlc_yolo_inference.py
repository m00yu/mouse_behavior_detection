import os
import numexpr
numexpr.set_num_threads(8)
import cv2
import numpy as np
import time
import yaml
import torch
import nidaqmx
from ultralytics import YOLO
from dlclive import DLCLive

# ---------------------- 설정 ----------------------
config_path_m1 = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/neurlab-dlc-models-m1/config_m1.yaml"
config_path_m2 = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/neurlab-dlc-models-m2/config_m2.yaml"

exported_model_m1 = os.path.join(os.path.dirname(config_path_m1), "exported-models", "DLC_TR_headplate_resnet_50_iteration-3_shuffle-1")
exported_model_m2 = os.path.join(os.path.dirname(config_path_m2), "exported-models", "DLC_TR_headplate_another_resnet_50_iteration-2_shuffle-1")

output_video_path = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/results/dlcdual_yolo_output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

keypoint_name = "nosetip"
individuals = ["m1", "m2"]

model_path = "C:/Users/NeuRLab/hyunsu/LEDdetect_yolo_realtime/runs/detect/train/weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)

DAQ_CHANNEL = "Dev2/ao0"
TTL_VOLTAGE = 5.0
TTL_DURATION = 1
ttl_cooldown = 2
last_ttl_time = 0

daq_task = nidaqmx.Task()
daq_task.ao_channels.add_ao_voltage_chan(DAQ_CHANNEL, min_val=-5.0, max_val=5.0)

dlc_m1 = DLCLive(exported_model_m1)
dlc_m2 = DLCLive(exported_model_m2)

def get_keypoint_index(config_path, keypoint_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    bodyparts = config.get("bodyparts", [])
    if keypoint_name not in bodyparts:
        raise ValueError(f"'{keypoint_name}' not found in {config_path}")
    return bodyparts.index(keypoint_name)

snout_idx_m1 = get_keypoint_index(config_path_m1, keypoint_name)
snout_idx_m2 = get_keypoint_index(config_path_m2, keypoint_name)

# ---------------------- 웹캠 ----------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 첫 프레임으로 DLC 초기화
ret, frame = cap.read()
if not ret:
    raise RuntimeError("프레임을 읽을 수 없습니다.")
dlc_m1.init_inference(frame)
dlc_m2.init_inference(frame)

# 튐 방지 변수
MAX_MOVE_DIST = 50  # 픽셀 단위
prev_pose_m1 = None
prev_pose_m2 = None

print("!!!시작!!!")
# ---------------------- 루프 ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_vis = frame.copy()

    # DLC 추론
    pose_m1 = dlc_m1.get_pose(frame)
    pose_m2 = dlc_m2.get_pose(frame)

    def draw_keypoints(pose, color, label_prefix, snout_idx, prev_pose):
        if pose is not None and pose.size > 0:
            if pose.ndim == 2:
                pose = np.expand_dims(pose, axis=0)

            filtered_pose = []
            for i, animal in enumerate(pose):
                valid_animal_pose = []
                for j, (x, y, p) in enumerate(animal):
                    if p < 0.5:
                        valid_animal_pose.append((None, None, p))
                        continue

                    if prev_pose is not None and i < len(prev_pose):
                        prev_x, prev_y, _ = prev_pose[i][j]
                        if prev_x is not None and prev_y is not None:
                            dist = np.linalg.norm([x - prev_x, y - prev_y])
                            if dist > MAX_MOVE_DIST:
                                valid_animal_pose.append((None, None, p))
                                continue

                    valid_animal_pose.append((x, y, p))
                    cv2.circle(frame_vis, (int(x), int(y)), 4, color, -1)
                    if j == snout_idx:
                        cv2.putText(frame_vis, f"{label_prefix}-snout", (int(x)+5, int(y)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                filtered_pose.append(valid_animal_pose)
            return filtered_pose
        return None

    filtered_pose_m1 = draw_keypoints(pose_m1, (0, 255, 0), "m1", snout_idx_m1, prev_pose_m1)
    filtered_pose_m2 = draw_keypoints(pose_m2, (0, 0, 255), "m2", snout_idx_m2, prev_pose_m2)

    prev_pose_m1 = filtered_pose_m1
    prev_pose_m2 = filtered_pose_m2

    # YOLO 추론
    results = model.predict(frame, device=device, verbose=False, imgsz=640)
    boxes_data = results[0].boxes.data

    best_box = None
    best_conf = 0.6
    best_cls_id = None

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
        cls_id = int(cls_id)
        if cls_id in [0, 1, 2, 3] and conf > best_conf:
            best_box = (x1, y1, x2, y2)
            best_conf = conf
            best_cls_id = cls_id

    if best_box is not None:
        now = time.time()

        lx1, ly1, lx2, ly2 = map(int, best_box)
        port_center = np.array([(lx1 + lx2) / 2, (ly1 + ly2) / 2])

        m1_nose = None
        m2_nose = None

        if filtered_pose_m1 and filtered_pose_m1[0][snout_idx_m1][:2] != (None, None):
            m1_nose = np.array(filtered_pose_m1[0][snout_idx_m1][:2])
        if filtered_pose_m2 and filtered_pose_m2[0][snout_idx_m2][:2] != (None, None):
            m2_nose = np.array(filtered_pose_m2[0][snout_idx_m2][:2])

        if m1_nose is not None and m2_nose is not None:
            dist_m1 = np.linalg.norm(m1_nose - port_center)
            dist_m2 = np.linalg.norm(m2_nose - port_center)

            if dist_m1 > dist_m2 and now - last_ttl_time > ttl_cooldown:
                print(f"🔔 m1 nose farther than m2 → TTL 신호 출력!")
                daq_task.write(TTL_VOLTAGE)
                time.sleep(TTL_DURATION)
                daq_task.write(0.0)
                last_ttl_time = now

        # draw YOLO box regardless of DAQ output
        label_map = {0: "bottom", 1: "left", 2: "right", 3: "up"}
        label = label_map.get(best_cls_id, f"port{best_cls_id}")
        cv2.rectangle(frame_vis, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
        cv2.putText(frame_vis, label, (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


    # 저장 및 시각화
    out.write(frame_vis)
    cv2.imshow("Pose + Port Detection", frame_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 종료합니다.")
        break

cap.release()
out.release()
daq_task.close()
cv2.destroyAllWindows()
print(f"✅ 저장 완료: {output_video_path}")
