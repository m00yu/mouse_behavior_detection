"""
python 코드로 DAQ output signal 보낼 수 있는지 확인하는 코드
"""

import nidaqmx
import time

voltages = [0.0, 1.0, 2.5, 5.0, 2.0, -1.0, 0.0]  # 원하는 전압 리스트
duration = 2  # 각 전압을 유지할 시간 (초)

with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan("Dev2/ao0", min_val=-5.0, max_val=5.0)
    
    for v in voltages:
        print(f"🔹 {v} V 출력 중...")
        task.write(v)
        time.sleep(duration)
