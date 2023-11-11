from inotify.adapters import Inotify
from mmaction.apis import inference_recognizer, init_recognizer
from threading import Thread
import pandas as pd
import time
import os
import torch
import json
import datetime

torch.set_num_threads(1)

CHECKPOINT = 'best_acc_top1_epoch_3.pth'
CONFIG = "mvit_koblik_224_16_5sec.py"
WATCH_DIR = "train_5s"
OUTPUT_FILE = "predicts.csv"
DEVICE = "cpu"

lbls_classes = {0: 'простой', 1: 'вынужденная', 2: 'сварка'}

classes_lbls = {'простой': 0, 'вынужденная': 1, 'сварка': 2}

target_names = ['простой', 'вынужденная', 'сварка']

model = init_recognizer(CONFIG, CHECKPOINT, device=DEVICE)
model.eval()


video_arr = []
last_zone_time = {}

zones = json.load(open('zones.json'))
zone_stats = {k: [] for k in zones.keys()}


# выгружает таблички
def loadout():
    global zone_stats
    try:
        for zone in zone_stats:
            zone_stats[zone].sort(key=lambda e: e['time'])

        empty_zones = False
        for zone in zone_stats:
            if len(zone_stats[zone]) == 0:
                empty_zones = True

        if empty_zones:
            return

        zone_labels = {}
        zone_lasts = {}

        column_data = {}
        data = []
        for zone in zone_stats:
            for element in zone_stats[zone]:
                del element['fname']
                data.append(element)

        for d in data:
            if not d['zone'] in zone_lasts:
                zone_lasts[d['zone']] = d
            if not d['zone'] in zone_labels:
                zone_labels[d['zone']] = {l: 0 for l in target_names}

            last = zone_lasts[d['zone']]
            zone_labels[d['zone']][last['label']] += d['time'] - last['time']
            zone_lasts[d['zone']] = d

        for d in data:
            d['time'] = str(datetime.datetime.fromtimestamp(d['time']))

        common_data = []

        for zone in zone_labels:
            sum_of_elements = sum(zone_labels[zone].values())
            if sum_of_elements == 0:
                continue
            for label in zone_labels[zone]:
                common_data.append(
                    {
                        'Зона': zone,
                        'Деятельность': label,
                        'Процент времени': zone_labels[zone][label]
                        / sum_of_elements
                        * 100,
                    }
                )

        df = pd.DataFrame(data)
        df.to_csv('result.csv')

        common_df = pd.DataFrame(common_data)
        common_df.to_csv('common_result.csv', index=False)

        zone_stats = {k: [] for k in zones.keys()}
    except ZeroDivisionError:
        pass


def loadout_thread():
    while True:
        # сохраняем раз в 10 минут
        time.sleep(600)
        loadout()


def process_video(video):
    while True:
        try:
            begin_time = time.time()
            predicted = inference_recognizer(model, video)
            print('done in', time.time() - begin_time, 's')
            break
        except RuntimeError:
            time.sleep(0.1)
    predicted_class = lbls_classes[int(predicted.pred_label.item())]
    return predicted_class


def process_video_thread():
    while True:
        for video in video_arr:
            if not os.path.exists(video['fname']):
                video_arr.remove(video)
                continue
            current_time = time.time()
            if current_time - video['time'] > 6:
                print(video['fname'])
                video['time'] = os.path.getmtime(video['fname'])
                label = process_video(video['fname'])
                print(label)
                video_arr.remove(video)
                video['label'] = label
                if (
                    len(zone_stats[video['zone']]) == 0
                    or zone_stats[video['zone']][-1]['label'] != label
                ):
                    zone_stats[video['zone']].append(video)
                    zone_stats[video['zone']].sort(key=lambda v: v['time'])
                os.remove(video['fname'])
        time.sleep(1)
        # print(zone_stats)


if __name__ == "__main__":
    # videos = glob(os.path.join(DATASET_DIR, "*.mp4"))

    _process_video_thread = Thread(target=process_video_thread, daemon=True)
    _process_video_thread.start()

    _loadout_thread = Thread(target=loadout_thread, daemon=True)
    _loadout_thread.start()

    names = []
    predicts = []
    gt = []

    inotify = Inotify()
    for zone in zones:
        inotify.add_watch(f'./{zone}')

    try:
        for event in inotify.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event

            if len(filename) == 0:
                continue
            if 'IN_CLOSE_WRITE' in type_names:
                video = os.path.join(path, filename)

                video_arr.append(
                    {'fname': video, 'time': time.time(), 'zone': path[2:]}
                )

    except KeyboardInterrupt:
        loadout()
