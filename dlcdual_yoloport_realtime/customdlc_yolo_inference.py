import os
import numexpr
numexpr.set_num_threads(8)
import cv2
import numpy as np
import time
import yaml
import torch
import nidaqmx
import threading
from ultralytics import YOLO
from dlclive import DLCLive

# ---------------------- 설정 ----------------------
config_path_m1 = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/neurlab-dlc-models-m1/config_m1.yaml"
config_path_m2 = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/neurlab-dlc-models-m2/config_m2.yaml"

exported_model_m1 = os.path.join(os.path.dirname(config_path_m1), "exported-models", "DLC_TR_headplate_resnet_50_iteration-3_shuffle-1")
exported_model_m2 = os.path.join(os.path.dirname(config_path_m2), "exported-models", "DLC_TR_headplate_another_resnet_50_iteration-2_shuffle-1")

output_video_path = "C:/Users/NeuRLab/hyunsu/dlcdual_yoloport_realtime/results/dlcdual_yolo_output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# 키포인트 이름 (DLC config에 실제 존재하는 값으로 맞춰야 함)
nose_key = "nosetip"
lheadplate_key = "Lheadplate"    # config 파일에 맞춰 수정
rheadplate_key = "Rheadplate"    # config 파일에 맞춰 수정

model_path = "C:/Users/NeuRLab/hyunsu/LEDdetect_yolo_realtime/runs/detect/train/weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)

DAQ_CHANNEL = "Dev2/ao0"
TTL_VOLTAGE = 5.0
TTL_DURATION = 1      # TTL 신호 지속시간 (초)
ttl_cooldown = 2      # TTL 신호 간 최소 간격 (초)
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

# 각 모델별 keypoint 인덱스 획득
nose_idx_m1 = get_keypoint_index(config_path_m1, nose_key)
lheadplate_idx_m1 = get_keypoint_index(config_path_m1, lheadplate_key)
rheadplate_idx_m1 = get_keypoint_index(config_path_m1, rheadplate_key)

nose_idx_m2 = get_keypoint_index(config_path_m2, nose_key)
lheadplate_idx_m2 = get_keypoint_index(config_path_m2, lheadplate_key)
rheadplate_idx_m2 = get_keypoint_index(config_path_m2, rheadplate_key)

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

# ---------------------- 변수 초기화 ----------------------
# 각 동물(마우스)의 마지막 유효 pose를 저장 (동물 인덱스를 키로 하는 dict)
# 각 값은 {'nose': (x, y), 'lhp': (x, y), 'rhp': (x, y)}
last_valid_pose_m1 = {}
last_valid_pose_m2 = {}

# nose tip에 대한 Kalman filter는 동물 인덱스를 키로 하는 dict로 저장
kalman_nose_m1 = {}
kalman_nose_m2 = {}

# 임계값 (픽셀 단위)
NOSE_JUMP_THRESHOLD = 150   # nose가 급격하게 튀었을 때 판단 기준
HEAD_DIST_THRESHOLD = 50    # nose와 headplate 간 최대 허용 거리

# ---------------------- Kalman filter helper 함수 ----------------------
def create_kalman_filter(initial_point):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePre = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)
    return kf

def update_kf(kf, measurement):
    pred = kf.predict()
    meas = np.array([[np.float32(measurement[0])],
                     [np.float32(measurement[1])]])
    corr = kf.correct(meas)
    return (corr[0, 0], corr[1, 0])

# ---------------------- nose 및 headplate 동시 검증 함수 ----------------------
def process_animal_pose(animal_pose, kalman_dict, last_valid_pose, nose_idx, lhp_idx, rhp_idx, frame_vis, label_prefix, color):
    """
    animal_pose: DLC 결과 (각 항목 (x, y, p))
    kalman_dict: 해당 동물의 nose에 대한 KF 객체 저장 (키: 'nose')
    last_valid_pose: 지금까지의 마지막 유효값 dict (키: 'nose', 'lhp', 'rhp')
    만약 현재 프레임의 측정치 중 하나라도 유효하지 않으면, last_valid_pose를 그대로 사용하고 이를 plot에 반영.
    """
    # 유효 여부 판단 함수
    def is_valid(meas):
        return (isinstance(meas, (list, tuple, np.ndarray)) and len(meas) >= 3 and
                meas[0] is not None and meas[1] is not None and meas[2] > 0.01)
    
    # 한 keypoint라도 현재 프레임에서 유효하지 않으면 last_valid_pose를 바로 사용
    if not (is_valid(animal_pose[nose_idx]) and is_valid(animal_pose[lhp_idx]) and is_valid(animal_pose[rhp_idx])):
        if last_valid_pose['nose'] is not None:
            return last_valid_pose
        else:
            return {'nose': (None, None), 'lhp': (None, None), 'rhp': (None, None)}
    
    # 현재 프레임의 후보값 추출
    nose_meas = animal_pose[nose_idx]
    lhp_meas = animal_pose[lhp_idx]
    rhp_meas = animal_pose[rhp_idx]
    
    # Nose 후보: Kalman filter 적용
    measurement = (nose_meas[0], nose_meas[1])
    if 'nose' not in kalman_dict:
        kalman_dict['nose'] = create_kalman_filter(measurement)
    updated_nose = update_kf(kalman_dict['nose'], measurement)
    
    # Headplate 후보: 단순 측정값 사용
    lhp_candidate = (lhp_meas[0], lhp_meas[1])
    rhp_candidate = (rhp_meas[0], rhp_meas[1])
    
    # headplate와 nose 간의 거리 조건 확인 함수
    def within_threshold(nose, head):
        return np.linalg.norm(np.array(nose) - np.array(head)) < HEAD_DIST_THRESHOLD
    
    valid_lhp = within_threshold(updated_nose, lhp_candidate)
    valid_rhp = within_threshold(updated_nose, rhp_candidate)
    
    # Nose jump 검사: last_valid_pose['nose']가 유효한지 체크 후 diff 계산
    if last_valid_pose['nose'] is not None and last_valid_pose['nose'][0] is not None:
        diff = np.linalg.norm(np.array(updated_nose) - np.array(last_valid_pose['nose']))
        if diff > NOSE_JUMP_THRESHOLD:
            updated_nose = last_valid_pose['nose']
            lhp_candidate = last_valid_pose['lhp']
            rhp_candidate = last_valid_pose['rhp']
            valid_lhp = valid_rhp = True

    # 만약 headplate 중 하나라도 유효하지 않으면 전체 롤백
    if not (valid_lhp and valid_rhp):
        updated_nose = last_valid_pose['nose']
        lhp_candidate = last_valid_pose['lhp']
        rhp_candidate = last_valid_pose['rhp']
    
    updated_pose = {'nose': updated_nose, 'lhp': lhp_candidate, 'rhp': rhp_candidate}
    
    # last_valid_pose 갱신: 만약 모든 값이 유효하면 갱신
    if (updated_pose['nose'][0] is not None and updated_pose['lhp'][0] is not None and updated_pose['rhp'][0] is not None):
        last_valid_pose.clear()
        last_valid_pose.update(updated_pose)
    
    # 시각화: last_valid_pose에 반영된 값 사용
    if updated_pose['nose'][0] is not None:
        cv2.circle(frame_vis, (int(updated_pose['nose'][0]), int(updated_pose['nose'][1])), 5, color, -1)
        cv2.putText(frame_vis, f"{label_prefix}-nose", (int(updated_pose['nose'][0]) + 5, int(updated_pose['nose'][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if updated_pose['lhp'][0] is not None:
        cv2.circle(frame_vis, (int(updated_pose['lhp'][0]), int(updated_pose['lhp'][1])), 5, color, -1)
        cv2.putText(frame_vis, f"{label_prefix}-lhead", (int(updated_pose['lhp'][0]) + 5, int(updated_pose['lhp'][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if updated_pose['rhp'][0] is not None:
        cv2.circle(frame_vis, (int(updated_pose['rhp'][0]), int(updated_pose['rhp'][1])), 5, color, -1)
        cv2.putText(frame_vis, f"{label_prefix}-rhead", (int(updated_pose['rhp'][0]) + 5, int(updated_pose['rhp'][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
    return updated_pose

# ---------------------- DAQ 신호 비동기 송신 함수 ----------------------
def send_TTL_pulse():
    daq_task.write(TTL_VOLTAGE)
    time.sleep(TTL_DURATION)
    daq_task.write(0.0)

# ---------------------- 메인 루프 ----------------------
print("!!!시작!!!")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    frame_vis = frame.copy()

    # DLC 추론
    pose_m1 = dlc_m1.get_pose(frame)
    pose_m2 = dlc_m2.get_pose(frame)
    
    # 단일 동물(2차원 배열)이면 리스트로 변환
    if pose_m1 is not None and hasattr(pose_m1, "ndim") and pose_m1.ndim == 2:
        pose_m1 = np.expand_dims(pose_m1, axis=0)
    if pose_m2 is not None and hasattr(pose_m2, "ndim") and pose_m2.ndim == 2:
        pose_m2 = np.expand_dims(pose_m2, axis=0)
    
    # 만약 DLC 결과가 없으면 None 처리
    if pose_m1 is None or (hasattr(pose_m1, 'size') and pose_m1.size == 0):
        pose_m1 = None
    if pose_m2 is None or (hasattr(pose_m2, 'size') and pose_m2.size == 0):
        pose_m2 = None

    # 각 모델별 동물(마우스) 처리
    processed_pose_m1 = []
    if pose_m1 is not None:
        for i, animal in enumerate(pose_m1):
            if i not in last_valid_pose_m1:
                last_valid_pose_m1[i] = {'nose': (None, None), 'lhp': (None, None), 'rhp': (None, None)}
            processed = process_animal_pose(
                animal,
                kalman_nose_m1.setdefault(i, {}),
                last_valid_pose_m1[i],
                nose_idx_m1, lheadplate_idx_m1, rheadplate_idx_m1,
                frame_vis, "m1", (0, 255, 0)
            )
            last_valid_pose_m1[i] = processed
            processed_pose_m1.append(processed)
    else:
        processed_pose_m1 = list(last_valid_pose_m1.values())

    processed_pose_m2 = []
    if pose_m2 is not None:
        for i, animal in enumerate(pose_m2):
            if i not in last_valid_pose_m2:
                last_valid_pose_m2[i] = {'nose': (None, None), 'lhp': (None, None), 'rhp': (None, None)}
            processed = process_animal_pose(
                animal,
                kalman_nose_m2.setdefault(i, {}),
                last_valid_pose_m2[i],
                nose_idx_m2, lheadplate_idx_m2, rheadplate_idx_m2,
                frame_vis, "m2", (0, 0, 255)
            )
            last_valid_pose_m2[i] = processed
            processed_pose_m2.append(processed)
    else:
        processed_pose_m2 = list(last_valid_pose_m2.values())

    # YOLO 추론 (예: LED detect 등)
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
        if processed_pose_m1 and processed_pose_m1[0]['nose'] != (None, None):
            m1_nose = np.array(processed_pose_m1[0]['nose'])
        m2_nose = None
        if processed_pose_m2 and processed_pose_m2[0]['nose'] != (None, None):
            m2_nose = np.array(processed_pose_m2[0]['nose'])
            
        if m1_nose is not None and m2_nose is not None:
            dist_m1 = np.linalg.norm(m1_nose - port_center)
            dist_m2 = np.linalg.norm(m2_nose - port_center)
            if dist_m1 > dist_m2 and now - last_ttl_time > ttl_cooldown:
                print("🔔 m1 nose farther than m2 → TTL 신호 출력!")
                threading.Thread(target=send_TTL_pulse, daemon=True).start()
                last_ttl_time = now

        label_map = {0: "bottom", 1: "left", 2: "right", 3: "up"}
        label = label_map.get(best_cls_id, f"port{best_cls_id}")
        cv2.rectangle(frame_vis, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
        cv2.putText(frame_vis, label, (lx1, ly1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
