mkdir train_5s

# ffmpeg -i tmp/nine_hour_left.mp4 -c copy -map 0 -segment_time 00:00:05 -f segment train_5s/nine_hour_left_%03d.mp4
ffmpeg -i tmp/nine_hour_right.mp4 -c copy -map 0 -segment_time 00:00:05 -f segment train_5s/nine_hour_right_%03d.mp4

ffmpeg -i tmp/five_hour_left.mp4 -c copy -map 0 -segment_time 00:00:05 -f segment train_5s/five_hour_left_%03d.mp4
ffmpeg -i tmp/five_hour_right.mp4 -c copy -map 0 -segment_time 00:00:05 -f segment train_5s/five_hour_right_%03d.mp4
