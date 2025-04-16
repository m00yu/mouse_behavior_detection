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

# 튐 방지 변수 및 smoothing 계수
MAX_MOVE_DIST = 100  # 픽셀 단위 – 큰 변화로 간주하는 기준
ALPHA = 0.3  # 0 (완전 이전값) ~ 1 (완전 현재값)의 보간 계수
prev_pose_m1 = None
prev_pose_m2 = None

print("!!!시작!!!")
# ---------------------- 루프 ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        # 프레임 누락 시, 짧은 시간 대기 후 continue (혹은 이전 프레임 활용)
        time.sleep(0.01)
        continue

    frame_vis = frame.copy()

    # DLC 추론
    pose_m1 = dlc_m1.get_pose(frame)
    pose_m2 = dlc_m2.get_pose(frame)

    # 추론 결과가 None 또는 빈 배열이면 이전 pose 사용
    if pose_m1 is None or (hasattr(pose_m1, 'size') and pose_m1.size == 0):
        pose_m1 = prev_pose_m1
    if pose_m2 is None or (hasattr(pose_m2, 'size') and pose_m2.size == 0):
        pose_m2 = prev_pose_m2

    # 삼각형 거리 기반 필터링 함수
    def validate_triangle(keypoints, triplet_idxs, ref_lengths, tol=0.3):
        """keypoints: [(x, y, p), ...] 형태의 리스트
        triplet_idxs: (i1, i2, i3)
        ref_lengths: (d12, d23, d13) 기준 거리
        tol: 허용 오차 비율"""
        try:
            p1 = np.array(keypoints[triplet_idxs[0]][:2])
            p2 = np.array(keypoints[triplet_idxs[1]][:2])
            p3 = np.array(keypoints[triplet_idxs[2]][:2])
            if None in p1 or None in p2 or None in p3:
                return False

            d12 = np.linalg.norm(p1 - p2)
            d23 = np.linalg.norm(p2 - p3)
            d13 = np.linalg.norm(p1 - p3)
            d12_ref, d23_ref, d13_ref = ref_lengths
            return (abs(d12 - d12_ref) / d12_ref < tol and
                    abs(d23 - d23_ref) / d23_ref < tol and
                    abs(d13 - d13_ref) / d13_ref < tol)
        except:
            return False

    # 초기 기준 거리 설정 함수
    def compute_ref_lengths(pose, idxs):
        try:
            p1 = np.array(pose[idxs[0]][:2])
            p2 = np.array(pose[idxs[1]][:2])
            p3 = np.array(pose[idxs[2]][:2])
            if None in p1 or None in p2 or None in p3:
                return None
            return (
                np.linalg.norm(p1 - p2),
                np.linalg.norm(p2 - p3),
                np.linalg.norm(p1 - p3)
            )
        except:
            return None

    # 보간(smoothing)을 적용한 키포인트 업데이트 함수
    def smooth_update(new_pt, prev_pt):
        # 이전 좌표와 현재 좌표가 모두 유효하면 선형 보간
        if new_pt is None or prev_pt is None:
            return new_pt if new_pt is not None else prev_pt
        return prev_pt * (1 - ALPHA) + new_pt * ALPHA

    # 키포인트 그리는 함수 (보간 적용)
    def draw_keypoints(pose, color, label_prefix, snout_idx, prev_pose, triangle_idxs, ref_lengths):
        # pose가 없으면 이전 pose 그대로 리턴
        if pose is None:
            return prev_pose
        if pose.size > 0:
            if pose.ndim == 2:
                pose = np.expand_dims(pose, axis=0)

            filtered_pose = []
            for i, animal in enumerate(pose):
                valid_animal_pose = []
                for j, (x, y, p) in enumerate(animal):
                    use_prev = False
                    # 낮은 confidence 또는 큰 이동이 있을 경우 보간 진행
                    if p < 0.1:
                        use_prev = True
                    elif prev_pose is not None and i < len(prev_pose):
                        prev_x, prev_y, _ = prev_pose[i][j]
                        if prev_x is not None and prev_y is not None:
                            dist = np.linalg.norm([x - prev_x, y - prev_y])
                            if dist > MAX_MOVE_DIST:
                                use_prev = True

                    if use_prev and prev_pose is not None and i < len(prev_pose):
                        # 보간: 이전값과 현재값을 선형 보간하여 부드러운 값 업데이트
                        prev_x, prev_y, prev_p = prev_pose[i][j]
                        new_x = smooth_update(x, prev_x) if prev_x is not None else x
                        new_y = smooth_update(y, prev_y) if prev_y is not None else y
                        combined_p = max(p, prev_p)
                        valid_animal_pose.append((new_x, new_y, combined_p))
                        cv2.circle(frame_vis, (int(new_x), int(new_y)), 4, color, -1)
                        if j == snout_idx:
                            cv2.putText(frame_vis, f"{label_prefix}-snout", (int(new_x)+5, int(new_y)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        valid_animal_pose.append((x, y, p))
                        cv2.circle(frame_vis, (int(x), int(y)), 4, color, -1)
                        if j == snout_idx:
                            cv2.putText(frame_vis, f"{label_prefix}-snout", (int(x)+5, int(y)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 삼각형 필터 검증 – 만약 검증에 실패하면 이전 pose 사용 (혹은 보간값 유지)
                if ref_lengths is not None and not validate_triangle(valid_animal_pose, triangle_idxs, ref_lengths):
                    if prev_pose is not None and i < len(prev_pose):
                        valid_animal_pose = prev_pose[i]
                filtered_pose.append(valid_animal_pose)
            return filtered_pose
        return prev_pose

    triplet_idxs = {
        "m1": (0, snout_idx_m1, 1),  # 예: 좌측 headplate, nosetip, 우측 headplate
        "m2": (0, snout_idx_m2, 1)
    }

    # 기준 거리 설정 (현재 프레임의 첫 동물의 pose에서)
    ref_lengths_m1 = None
    ref_lengths_m2 = None
    try:
        if pose_m1 is not None:
            ref_lengths_m1 = compute_ref_lengths(pose_m1[0], triplet_idxs["m1"])
        if pose_m2 is not None:
            ref_lengths_m2 = compute_ref_lengths(pose_m2[0], triplet_idxs["m2"])
    except:
        pass

    filtered_pose_m1 = draw_keypoints(pose_m1, (0, 255, 0), "m1", snout_idx_m1, prev_pose_m1, triplet_idxs["m1"], ref_lengths_m1)
    filtered_pose_m2 = draw_keypoints(pose_m2, (0, 0, 255), "m2", snout_idx_m2, prev_pose_m2, triplet_idxs["m2"], ref_lengths_m2)

    # 업데이트 전에 null 체크
    if filtered_pose_m1 is not None:
        prev_pose_m1 = filtered_pose_m1
    if filtered_pose_m2 is not None:
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

        # YOLO 박스 그리기
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
