import os

import numpy as np
from sklearn.metrics import confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from glob import glob

import ffmpeg

import warnings
warnings.filterwarnings('ignore')


os.makedirs('tmp', exist_ok=True)


input_file = ffmpeg.input('Коблик/nine_hour_video/Video.mp4')
output_file = ffmpeg.output(input_file.filter('scale', width='-1', height='224'), 'tmp/nine_hour_video_224.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)


input_file = ffmpeg.input('Коблик/five_hour_video/Video.mp4')
output_file = ffmpeg.output(input_file.filter('scale', width='-1', height='224'), 'tmp/five_hour_video_224.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)



crop_dimensions = '199:224:0:0'
input_file = ffmpeg.input('tmp/nine_hour_video_224.mp4')
output_file = ffmpeg.output(input_file.filter('crop', *crop_dimensions.split(':')), 'tmp/nine_hour_left.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)

crop_dimensions = '199:224:199:0'
input_file = ffmpeg.input('tmp/nine_hour_video_224.mp4')
output_file = ffmpeg.output(input_file.filter('crop', *crop_dimensions.split(':')), 'tmp/nine_hour_right.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)


crop_dimensions = '199:224:0:0'
input_file = ffmpeg.input('tmp/five_hour_video_224.mp4')
output_file = ffmpeg.output(input_file.filter('crop', *crop_dimensions.split(':')), 'tmp/five_hour_left.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)

crop_dimensions = '199:224:199:0'
input_file = ffmpeg.input('tmp/five_hour_video_224.mp4')
output_file = ffmpeg.output(input_file.filter('crop', *crop_dimensions.split(':')), 'tmp/five_hour_right.mp4')
ffmpeg.run(output_file, quiet=True, overwrite_output=True)
