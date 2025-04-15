"""
input: video
output: multi-animal pose estimation + water port detection

DLC multi-animal 기능으로 multi-animal pose estimation
YOLO로 water port detection
"""
import os
import cv2
import numpy as np
import torch
import deeplabcut as dlc
import pandas as pd
from ultralytics import YOLO

# --------------------- DLC Config 반영 ---------------------
project_path = "C:/Users/NeuRLab/hyunsu/dlcmulti_yoloport_video"
config_path = os.path.join(project_path, "config.yaml")
video_name = "short_short_Top.mp4"
video_path = os.path.join(project_path, "videos", video_name)

# DLC 개체명 & keypoint 설정
individuals = ["mus1", "mus2"]
keypoint_name = "snout"  # 코로 머리 방향

# --------------------- YOLO 모델 로드 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('C:/Users/NeuRLab/hyunsu/model/yolo_port.pt')
model.to(device)

# --------------------- YOLO를 이용한 불빛 위치 탐색 ---------------------
cap = cv2.VideoCapture(video_path)
print("영상 열기 성공 여부:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

light_detections = {}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델 예측
    results = model.predict(frame, device=device, verbose=False)
    boxes_data = results[0].boxes.data

    # 가장 신뢰도 높은 불빛 bbox 저장
    light_box = None
    light_conf = 0.0
    light_cls_id = None

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box[:6].tolist()
        cls_id = int(cls_id)

        if cls_id in [0, 1, 2, 3]:
            if conf > light_conf:
                light_box = (x1, y1, x2, y2)
                light_conf = conf
                light_cls_id = cls_id

    if light_box is not None:
        light_detections[frame_idx] = (light_box, light_cls_id)

    frame_idx += 1

cap.release()
print(f"YOLO light detection completed: {len(light_detections)} frames processed.")

# --------------------- DLC 분석 수행 ---------------------
print("Running DLC analysis...")
dlc.analyze_videos(config_path, [video_path], shuffle=0, videotype="mp4", auto_track=False)

# DLC로 탐지된 keypoints를 tracklet으로 변환
dlc.convert_detections2tracklets(
    config_path,
    [video_path],
    videotype="mp4",
    shuffle=0,
    track_method="ellipse",
    ignore_bodyparts=["tail1", "tail2", "tailend"],
)

# DLC의 tracklet 결과 보정
dlc.stitch_tracklets(
    config_path,
    [video_path],
    videotype="mp4",
    shuffle=0,
    track_method="ellipse",
    n_tracks=2,
)

# DLC 예측 결과 필터링
dlc.filterpredictions(
    config_path,
    [video_path],
    shuffle=0,
    videotype="mp4",
    track_method="ellipse",
)

# DLC 라벨링된 비디오 생성
print("Creating DLC labeled video...")
dlc.create_labeled_video(
    config_path,
    [video_path],
    videotype="mp4",
    shuffle=0,
    color_by="individual",
    keypoints_only=False,
    draw_skeleton=True,
    filtered=True,
    track_method="ellipse"
)

# --------------------- DLC 분석 결과 로드 (HDF5 파일) ---------------------
h5_file = os.path.join(project_path, "videos", f"{video_name.split('.mp4')[0]}DLC_dlcrnetms5_demoJul14shuffle0_20000_el_filtered.h5")

print(f"Loading DLC keypoints from {h5_file}...")
df = pd.read_hdf(h5_file, key='df_with_missing')  
scorer = df.columns.levels[0][0]

# --------------------- YOLO 결과를 DLC 라벨링된 비디오에 추가 ---------------------
print("Merging YOLO results with DLC labeled video...")
labeled_video_path = video_path.replace(".mp4", "DLC_dlcrnetms5_demoJul14shuffle0_20000_el_filtered_id_labeled.mp4")
cap = cv2.VideoCapture(labeled_video_path)

output_video_path = os.path.join(project_path, "videos", "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
prev_positions = {ind: None for ind in individuals}  # 이전 프레임에서의 머리 위치 저장

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO에서 저장한 port bbox 가져오기
    if frame_idx in light_detections:
        light_box, light_cls_id = light_detections[frame_idx]
        lx1, ly1, lx2, ly2 = map(int, light_box)
        light_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

        # port 위치
        label_map = {0: "light_bottom", 1: "light_left", 2: "light_right", 3: "light_up"}
        light_label = label_map.get(light_cls_id, "light")
        cv2.putText(frame, light_label, (lx1, ly1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # DLC에서 keypoints 가져오기
        for ind in individuals:
            col_x = (scorer, ind, keypoint_name, "x")
            col_y = (scorer, ind, keypoint_name, "y")

            if col_x in df.columns and col_y in df.columns:
                x, y = df.at[frame_idx, col_x], df.at[frame_idx, col_y]
                if not np.isnan(x) and not np.isnan(y):
                    prev_positions[ind] = (int(x), int(y))

        # 불빛과 가까운 마우스 판단
        if all(prev_positions.values()):
            dist_m1 = np.linalg.norm(np.array(prev_positions["mus1"]) - np.array(light_center))
            dist_m2 = np.linalg.norm(np.array(prev_positions["mus2"]) - np.array(light_center))

            closest_mouse = "mus1" if dist_m1 < dist_m2 else "mus2"
            cv2.putText(frame, f"{closest_mouse} closest to {light_label}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 최종 결과 비디오 저장
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Final video with DLC + YOLO + closest mouse saved at {output_video_path}")
