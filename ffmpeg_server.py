import json
import subprocess
import shlex

zones = json.load(open('zones.json'))

threads: list[subprocess.Popen] = []

for zone_name in zones:
    zone_data = zones[zone_name]
    side = zone_data['side']

    crop = "iw/2:ih:ow:0" if side == 'right' else 'iw/2:ih:0:0'

    func = f"ffmpeg -f segment -segment_time 5 -reset_timestamps 1 -filter_complex \"[0]crop={crop}[{side}]\" -map \"[{side}]\" \"{zone_name}/{side}%03d.mp4\" -i rtsp://localhost:8554/mystream"
    print(func)
    proc = subprocess.Popen(shlex.split(func))
    threads.append(proc)


for thread in threads:
    thread.communicate()
