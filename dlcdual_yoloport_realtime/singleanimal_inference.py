"""
mouse 한마리 일때만
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import tensorflow as tf

from dlclive import DLCLive
import yaml

# --------------------- 환경 및 경로 설정 ---------------------
# PyTorch용 YOLO는 첫 번째 노출된 GPU (즉, 물리 GPU 4)를 사용
device = torch.device("cuda")  

# DLC 프로젝트 및 config 경로
project_path = "/hdd/hyunsu/mouse/yolo_detect_light/dlc/demo-me-2021-07-14"  # export된 경로
config_path = os.path.join(project_path, "config.yaml")
video_name = "short_short_Top.mp4"
video_path = os.path.join(project_path, "videos", video_name)

# DLC 대상 개체와 관심 keypoint (거리 계산용 'snout')
individuals = ["mus1", "mus2"]
keypoint_name = "snout"

# --------------------- YOLO 모델 로드 (PyTorch, GPU 4) ---------------------
model = YOLO('/hdd/hyunsu/mouse/yolo_detect_light/runs/detect/train/weights/best.pt')
model.to(device)

# --------------------- DLCLive 로드 (내부적으로 TensorFlow가 GPU 7 사용) ---------------------
exported_model_path = os.path.join(project_path, "exported-models", "DLC_demo_resnet_50_iteration-0_shuffle-0")
dlc_live = DLCLive(exported_model_path)

# config 파일에서 bodyparts 순서를 확인 (전체 keypoint 이름 목록)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
bodyparts = config.get("bodyparts", [])
if keypoint_name not in bodyparts:
    raise ValueError(f"Keypoint '{keypoint_name}' not found in config bodyparts.")
snout_index = bodyparts.index(keypoint_name)

# --------------------- 비디오 캡처 및 출력 설정 ---------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = os.path.join(project_path, "videos", "output_video_realtime.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

test_img_folder = os.path.join(project_path, "videos", "test_frames")
os.makedirs(test_img_folder, exist_ok=True)

frame_idx = 0

# --------------------- 초기 프레임 읽기 및 DLC 초기화 ---------------------
ret, first_frame = cap.read()
if not ret:
    cap.release()
    out.release()
    raise RuntimeError("비디오를 읽어올 수 없습니다.")

# 첫 프레임을 이용해 DLC 추론 초기화 (한 번만 호출)
dlc_live.init_inference(first_frame)

# 첫 프레임 처리 (동일 코드 진행)
frame = first_frame.copy()

# YOLO: 불빛 객체 탐지 (첫 프레임)
results = model.predict(frame, device=str(device), verbose=False)
boxes_data = results[0].boxes.data.cpu().numpy()
light_box = None
light_conf = 0.0
light_cls_id = None

for box in boxes_data:
    x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
    cls_id = int(cls_id)
    if cls_id in [0, 1, 2, 3] and conf > light_conf:
        light_box = (int(x1), int(y1), int(x2), int(y2))
        light_conf = conf
        light_cls_id = cls_id

if light_box is not None:
    lx1, ly1, lx2, ly2 = light_box
    light_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
    label_map = {0: "light_bottom", 1: "light_left", 2: "light_right", 3: "light_up"}
    light_label = label_map.get(light_cls_id, "light")
    cv2.putText(frame, light_label, (lx1, ly1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
else:
    light_center = None
    light_label = "none"

# DLC: 첫 프레임 keypoint 예측 및 모든 keypoint 표시 (점만 표시)
keypoints = dlc_live.get_pose(first_frame)
positions = {}  # snout 좌표 저장 (거리 계산용)
if keypoints is not None and keypoints.size > 0:
    if keypoints.ndim == 2:
        keypoints = np.expand_dims(keypoints, axis=0)
    for i, animal in enumerate(keypoints):
        for j in range(animal.shape[0]):
            x, y, likelihood = animal[j]
            if likelihood > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0) if i == 0 else (0, 255, 0), -1)
                if j == snout_index:
                    positions[individuals[i]] = (int(x), int(y))
else:
    for ind in individuals:
        positions[ind] = None

if light_center is not None and all(positions.get(ind) is not None for ind in individuals):
    dists = {}
    for ind in individuals:
        pos = np.array(positions[ind])
        dists[ind] = np.linalg.norm(pos - np.array(light_center))
    closest_mouse = min(dists, key=dists.get)
    cv2.putText(frame, f"{closest_mouse} closest to {light_label}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# 첫 프레임 결과 저장 (비디오와 테스트 이미지)
out.write(frame)
cv2.imwrite(os.path.join(test_img_folder, f"frame_{frame_idx}.jpg"), frame)
frame_idx += 1

# --------------------- 나머지 프레임 처리 ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.time()
    # YOLO: 불빛 객체 탐지
    results = model.predict(frame, device=str(device), verbose=False)
    boxes_data = results[0].boxes.data.cpu().numpy()
    light_box = None
    light_conf = 0.0
    light_cls_id = None

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
        cls_id = int(cls_id)
        if cls_id in [0, 1, 2, 3] and conf > light_conf:
            light_box = (int(x1), int(y1), int(x2), int(y2))
            light_conf = conf
            light_cls_id = cls_id

    if light_box is not None:
        lx1, ly1, lx2, ly2 = light_box
        light_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
        label_map = {0: "light_bottom", 1: "light_left", 2: "light_right", 3: "light_up"}
        light_label = label_map.get(light_cls_id, "light")
        cv2.putText(frame, light_label, (lx1, ly1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        light_center = None
        light_label = "none"

    # DLC: 프레임 단위 keypoint 예측 (점만 표시)
    keypoints = dlc_live.get_pose(frame)
    positions = {}  # snout 좌표 저장용
    if keypoints is not None and keypoints.size > 0:
        if keypoints.ndim == 2:
            keypoints = np.expand_dims(keypoints, axis=0)
        for i, animal in enumerate(keypoints):
            for j in range(animal.shape[0]):
                x, y, likelihood = animal[j]
                if likelihood > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0) if i == 0 else (0, 255, 0), -1)
                    if j == snout_index:
                        positions[individuals[i]] = (int(x), int(y))
    else:
        for ind in individuals:
            positions[ind] = None

    if light_center is not None and all(positions.get(ind) is not None for ind in individuals):
        dists = {}
        for ind in individuals:
            pos = np.array(positions[ind])
            dists[ind] = np.linalg.norm(pos - np.array(light_center))
        closest_mouse = min(dists, key=dists.get)
        cv2.putText(frame, f"{closest_mouse} closest to {light_label}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    end = time.time()
    # print(end-start)
    out.write(frame)
    
    # test_img_path = os.path.join(test_img_folder, f"frame_{frame_idx}.jpg")
    # cv2.imwrite(test_img_path, frame)
    
    frame_idx += 1

cap.release()
out.release()
print(f"Output video saved at {output_video_path}")
