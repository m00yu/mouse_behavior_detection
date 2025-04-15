import os
import cv2
import numpy as np
import time
import yaml

from dlclive import DLCLive


# ---------------------- 설정 ----------------------
# 경로 설정 (config.yaml, exported-models 등)
config_path_m1 = "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/neurlab-dlc-models-m1/config_m1.yaml"
config_path_m2 = "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/neurlab-dlc-models-m2/config_m2.yaml"

exported_model_m1 = os.path.join(os.path.dirname(config_path_m1), "exported-models", "DLC_TR_headplate_resnet_50_iteration-3_shuffle-1")
exported_model_m2 = os.path.join(os.path.dirname(config_path_m2), "exported-models", "DLC_TR_headplate_another_resnet_50_iteration-2_shuffle-1")

output_video_path = "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/results/dlclive_output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# 관심 keypoint 이름 (예: 'snout' 또는 'nose')
keypoint_name = "nosetip"
individuals = ["m1", "m2"]


# ---------------------- DLC 모델 로드 ----------------------
dlc_m1 = DLCLive(exported_model_m1)
dlc_m2 = DLCLive(exported_model_m2)

# config에서 keypoint 인덱스 찾기
def get_keypoint_index(config_path, keypoint_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    bodyparts = config.get("bodyparts", [])
    if keypoint_name not in bodyparts:
        raise ValueError(f"'{keypoint_name}' not found in {config_path}")
    return bodyparts.index(keypoint_name)

snout_idx_m1 = get_keypoint_index(config_path_m1, keypoint_name)
snout_idx_m2 = get_keypoint_index(config_path_m2, keypoint_name)


# ---------------------- 실시간 웹캠 처리 ----------------------
cap = cv2.VideoCapture(0)  # 0번 카메라 (웹캠)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 초기화: 첫 프레임 읽고 DLC 모델 초기화
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("프레임을 읽을 수 없습니다.")

dlc_m1.init_inference(frame)
dlc_m2.init_inference(frame)

frame_idx = 0

# ---------------------- 프레임 반복 ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_vis = frame.copy()

    # DLC 추론 (각 모델별)
    pose_m1 = dlc_m1.get_pose(frame)
    pose_m2 = dlc_m2.get_pose(frame)

    # 결과 시각화
    def draw_keypoints(pose, color, label_prefix, snout_idx):
        if pose is not None and pose.size > 0:
            if pose.ndim == 2:
                pose = np.expand_dims(pose, axis=0)
            for i, animal in enumerate(pose):
                for j, (x, y, p) in enumerate(animal):
                    if p > 0.5:
                        cv2.circle(frame_vis, (int(x), int(y)), 4, color, -1)
                        if j == snout_idx:
                            cv2.putText(frame_vis, f"{label_prefix}-snout", (int(x)+5, int(y)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_keypoints(pose_m1, (0, 255, 0), "m1", snout_idx_m1)  # 초록색
    draw_keypoints(pose_m2, (0, 0, 255), "m2", snout_idx_m2)  # 빨간색

    # 결과 저장 및 출력
    out.write(frame_vis)
    cv2.imshow("Dual DLC Live", frame_vis)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("종료합니다.")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 결과 저장 완료: {output_video_path}")
