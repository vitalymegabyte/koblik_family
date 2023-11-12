import os

from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


CHECKPOINT = 'best_acc_top1_epoch_4.pth'
CONFIG = "mvit_koblik_224_24_5sec.py"
DATASET_DIR = "train_5s"
OUTPUT_FILE = "predicts.csv"
DEVICE = "cpu"

lbls_classes = {0: 'простой', 1: 'вынужденная', 2: 'сварка'}

classes_lbls = {'простой': 0, 'вынужденная': 1, 'сварка': 2}

target_names = ['простой', 'вынужденная', 'сварка']


if __name__ == "__main__":
    df_nine_left = pd.read_csv('nine_hour_left.csv')
    df_nine_right = pd.read_csv('nine_hour_right.csv')
    df_five_left = pd.read_csv('five_hour_left.csv')
    df_five_right = pd.read_csv('five_hour_right.csv')

    # videos = glob(os.path.join(DATASET_DIR, "*.mp4"))

    model = init_recognizer(CONFIG, CHECKPOINT, device=DEVICE)
    model.eval()

    names = []
    predicts = []
    gt = []

    for _, r in tqdm(df_nine_left[df_nine_left['time'] <= 7200].iterrows()):
        # break
        video = os.path.join(DATASET_DIR, r['fname'])
        name = os.path.basename(video).replace(".mp4", "")
        names.append(name)
        predicted = inference_recognizer(model, video)
        predicted_class = int(predicted.pred_label.item())
        predicts.append(predicted_class)
        gt.append(classes_lbls[r['label']])

    for _, r in tqdm(df_nine_right[df_nine_right['time'] <= 7200].iterrows()):
        # break
        video = os.path.join(DATASET_DIR, r['fname'])
        name = os.path.basename(video).replace(".mp4", "")
        names.append(name)
        predicted = inference_recognizer(model, video)
        predicted_class = int(predicted.pred_label.item())
        predicts.append(predicted_class)
        gt.append(classes_lbls[r['label']])

    for _, r in tqdm(df_five_left[df_five_left['time'] >= 13845].iterrows()):
        # break
        video = os.path.join(DATASET_DIR, r['fname'])
        name = os.path.basename(video).replace(".mp4", "")
        names.append(name)
        predicted = inference_recognizer(model, video)
        predicted_class = int(predicted.pred_label.item())
        predicts.append(predicted_class)
        gt.append(classes_lbls[r['label']])

    for _, r in tqdm(df_five_right[df_five_right['time'] >= 13845].iterrows()):
        # break
        video = os.path.join(DATASET_DIR, r['fname'])
        name = os.path.basename(video).replace(".mp4", "")
        names.append(name)
        predicted = inference_recognizer(model, video)
        predicted_class = int(predicted.pred_label.item())
        predicts.append(predicted_class)
        gt.append(classes_lbls[r['label']])

    result_df = pd.DataFrame.from_dict({"video_id": names, "class_indx": predicts})
    result_df['label'] = result_df['class_indx'].map(lbls_classes)
    result_df['gt_class_indx'] = gt
    result_df['gt_label'] = result_df['gt_class_indx'].map(lbls_classes)

    result_df.to_csv(OUTPUT_FILE, index=False)

    print(
        classification_report(
            result_df['gt_class_indx'],
            result_df['class_indx'],
            target_names=target_names,
            digits=4,
        )
    )

    # matrix = confusion_matrix(result_df['gt_class_indx'], result_df['class_indx'])
    # matrix.diagonal()/matrix.sum(axis=1)

    ConfusionMatrixDisplay.from_predictions(result_df['gt_label'], result_df['label'])
