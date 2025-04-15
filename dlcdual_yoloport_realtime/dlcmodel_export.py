"""
DLC-live를 쓰려면 DLC 모델을 export 해서 
pose_cfg.yaml, *.pb 파일을 만들어야 함
<project_path>/exported-models 폴더 만들어짐
"""

import deeplabcut

config_path = "C:/Users/NeuRLab/hyunsu/dlcsingle_yoloport_realtime/neurlab-dlc-models-m2/config_m2.yaml"
deeplabcut.export_model(config_path)