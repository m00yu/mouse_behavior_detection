"""
YOLO 없이 DLC만 -> DLC 잘 작동하는지 확인

python customdlc_video_tmp.py
    --video_path "C:/Users/NeuRLab/hyunsu/video/short_short_Top.mp4"
    --config_m1 "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/neurlab-dlc-models-m1/config_m1.yaml"
    --config_m2 "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/neurlab-dlc-models-m2/config_m2.yaml"
    --output_dir "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/results"
    --individual1 "m1"
    --individual2 "m2"

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import deeplabcut
import pandas as pd
import numpy as np
import cv2
from deeplabcut.utils import auxiliaryfunctions

# -----------------------------------------------------------------------------
# 1. Logger Class
# -----------------------------------------------------------------------------
class Logger:
    def __init__(self, logfile):
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.logger = logging.getLogger('DualSingleDLC')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
        file_handler = logging.FileHandler(logfile + '.log', mode='a')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, msg, *args):
        self.logger.info(msg, *args)

    def error(self, msg, *args):
        self.logger.error(msg, *args)

# -----------------------------------------------------------------------------
# 2. Run Single-Animal DLC Analysis for each config
# -----------------------------------------------------------------------------
def analyze_video_single(config_path, video_path, output_dir, logger):
    """
    Run DLC analyze_videos for a single-animal config and return the path of the CSV/H5 result.
    """
    # Read config
    cfg = auxiliaryfunctions.read_config(config_path)
    cfg['project_path'] = os.path.dirname(config_path)  # Ensure project_path is set properly

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Analyzing with config: {config_path}")
    deeplabcut.analyze_videos(
        config_path,
        [video_path],
        gputouse=0,           # GPU index, if needed
        save_as_csv=True,
        destfolder=output_dir
    )
    logger.info(f"Files in {output_dir} after analysis: {os.listdir(output_dir)}")
    # DLC가 만드는 CSV/H5 파일 이름 규칙
    # 대개: <videoName>...csv or <videoName>...h5 형태
    # 정확한 파일명을 찾기 위해 결과물 후보를 스캔하거나 DLC 소스 함수를 참조
    # 간단히, output_dir 안에서 가장 최근에 생성된 CSV를 반환한다고 가정.
    csv_path = find_newest_csv(output_dir, os.path.basename(video_path))
    logger.info(f"Result CSV: {csv_path}")

    return csv_path

def find_newest_csv(folder, video_basename):
    # video_basename = "short_Top.mp4"
    # -> stem = "short_Top"
    stem = os.path.splitext(video_basename)[0]

    csv_files = [
        f for f in os.listdir(folder) 
        if f.endswith('.csv') and stem in f
    ]
    if not csv_files:
        return None
    csv_files = sorted(csv_files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    newest_csv = csv_files[-1]
    return os.path.join(folder, newest_csv)

# -----------------------------------------------------------------------------
# 3. Merge the two CSV results
# -----------------------------------------------------------------------------
def merge_two_dlc_csv(csv_m1, csv_m2, logger, 
                      individual1="m1", individual2="m2"):
    """
    두 개의 Single-Animal DLC CSV 파일을 하나의 DataFrame으로 병합.
    bodyparts가 같은지 확인. 
    - m1, m2 구분을 위해 컬럼에 prefix를 붙인다.
    """
    logger.info(f"Merging CSV: \n  - {csv_m1}\n  - {csv_m2}")
    
    df_m1 = pd.read_csv(csv_m1, header=[0,1,2], index_col=0)
    df_m2 = pd.read_csv(csv_m2, header=[0,1,2], index_col=0)
    
    # DLC CSV는 멀티-헤더 구조:
    # Level 0: Scorer
    # Level 1: Bodyparts
    # Level 2: coords (x, y, likelihood)
    
    # m1, m2 각각 prefix 달기
    # 예: ('DLC_resnet50...', 'nose', 'x') -> ('m1', 'nose', 'x')
    
    new_columns_m1 = []
    for col in df_m1.columns:
        # col은 tuple 형태: (scorer, bodypart, coord)
        new_columns_m1.append((individual1,) + col[1:])  # (m1, bodypart, x/y/likelihood)
        
    new_columns_m2 = []
    for col in df_m2.columns:
        new_columns_m2.append((individual2,) + col[1:])
    
    df_m1.columns = pd.MultiIndex.from_tuples(new_columns_m1, names=["individual", "bodypart", "coords"])
    df_m2.columns = pd.MultiIndex.from_tuples(new_columns_m2, names=["individual", "bodypart", "coords"])
    
    # 인덱스(프레임) 같다고 가정하고 단순 병합(concat)
    # 만약 frame 수가 다르면, 공통 구간이 아닌 곳은 NaN
    df_merged = pd.concat([df_m1, df_m2], axis=1)
    return df_merged

# -----------------------------------------------------------------------------
# 4. Create a combined labeled video using OpenCV
# -----------------------------------------------------------------------------
def create_combined_labeled_video(video_path, df_merged, output_path, logger,
                                  marker_size=5, thickness=2):
    """
    df_merged (MultiIndex: [individual, bodypart, coords])를 참조하여
    각 프레임마다 m1, m2의 bodypart 위치를 그린 영상을 만든다.
    
    - video_path: 원본 동영상 경로
    - df_merged: 병합된 DLC 결과 (index=frame, columns MultiIndex)
    - output_path: 저장할 영상 경로
    """
    logger.info(f"Creating combined labeled video -> {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 준비: 멀티인덱스 구조 파악
    individuals = df_merged.columns.levels[0]  # m1, m2...
    bodyparts = df_merged.columns.levels[1]    # e.g. nose, ear, ...
    coords = df_merged.columns.levels[2]       # x, y, likelihood
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in df_merged.index:
            # 각 individual x,y coord 읽어서 프레임에 표시
            for indiv in individuals:
                for bp in bodyparts:
                    x_col = (indiv, bp, 'x')
                    y_col = (indiv, bp, 'y')
                    l_col = (indiv, bp, 'likelihood')
                    
                    if x_col in df_merged.columns and y_col in df_merged.columns:
                        x = df_merged.loc[frame_idx, x_col]
                        y = df_merged.loc[frame_idx, y_col]
                        l = df_merged.loc[frame_idx, l_col] if l_col in df_merged.columns else 1.0
                        
                        if not np.isnan(x) and not np.isnan(y) and l > 0.2: # 임계 likelihood=0.2 예시
                            # 점 찍기
                            # ※ 색상은 m1, m2 구분해서 다르게 하고 싶으면 다양한 방법 가능
                            #    여기서는 임의로 indiv에 따라 다르게: m1=green, m2=blue (예시)
                            color = (0, 255, 0) if indiv == 'm1' else (255, 0, 0)
                            cv2.circle(frame, (int(x), int(y)), marker_size, color, thickness)
                            # bodypart 텍스트 표기
                            cv2.putText(frame, f"{indiv}-{bp}", (int(x)+5, int(y)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    logger.info("Video generation complete.")

# -----------------------------------------------------------------------------
# 5. Main function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run two Single-Animal DLC configs, merge results, and create combined video.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--config_m1', type=str, required=True, help='Path to config_m1.yaml')
    parser.add_argument('--config_m2', type=str, required=True, help='Path to config_m2.yaml')
    parser.add_argument('--output_dir', type=str, default='./dlc_dual_output', help='Directory to save merged results and video')
    parser.add_argument('--individual1', type=str, default='m1', help='Name/prefix for the first animal')
    parser.add_argument('--individual2', type=str, default='m2', help='Name/prefix for the second animal')
    
    args = parser.parse_args()
    
    logger = Logger(os.path.join(args.output_dir, 'dual_dlc_log'))
    logger.info("===== Start Dual Single-Animal DLC Pipeline =====")
    
    # 1) Analyze with config_m1
    csv_m1 = analyze_video_single(args.config_m1, args.video_path, args.output_dir, logger)
    
    # 2) Analyze with config_m2
    csv_m2 = analyze_video_single(args.config_m2, args.video_path, args.output_dir, logger)
    
    # 3) Merge two CSV
    df_merged = merge_two_dlc_csv(csv_m1, csv_m2, logger, 
                                  individual1=args.individual1, 
                                  individual2=args.individual2)
    
    merged_csv_path = os.path.join(args.output_dir, "merged_dlc_results.csv")
    df_merged.to_csv(merged_csv_path)
    logger.info(f"Saved merged CSV: {merged_csv_path}")
    
    # 4) Create combined labeled video using the merged DataFrame
    combined_video_path = os.path.join(args.output_dir, "combined_labeled_video.mp4")
    create_combined_labeled_video(args.video_path, df_merged, combined_video_path, logger)
    
    logger.info("===== All Done! =====")

if __name__ == "__main__":
    main()
