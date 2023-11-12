# Решение кейса "KOBLiK"

Решение представляет собой три микросервиса: 

- `ffmpeg_server.py` -- служебный контейнер для разбивки потоков на зоны и отрезки по 5 секунд. Результаты деятельности он складывает в папку
- `video_predictor.py` -- контейнер, в котором крутится нейронка. Предсказывает класс работы, проводящейся в зоне
- `bot.py` -- собственно бот, интерфейс для взаимодействия

Для запуска всего вместе используется скрипт `run.sh`

## Конфигурация

Для того, чтобы добавить/удалить зоны можно поменять файл `zones.json`. По сути -- это главный конфиг решения. Формат у него следующий:

```
{
    "Название зоны": {
        "rtspstream": "ссылка на поток по протоколу rtsp",
        "side": "сторона с которой расположена зона на камере: левая или правая"
    },
}
```

## Работа под нагрузкой

Во время работы решение потребляет 2 ядра CPU на одну камеру (по одному на зону) для инференса, и ещё 1 ядро для разбивки потока. Необходимо это учитывать при выборе мощностей для запуска

## Веса модели

Веса модели загружены в релиз: https://github.com/vitalymegabyte/koblik_family/releases/tag/weights

## Скрипты для обучения модели в каталоге train_scripts.

1. Подготовка данных
 - В каталог "Коблик" распаковать исходные данные (nine_hour_video и five_hour_video).
 - python prepare_dataset.py   (Уменьшит размер до 224*398, разрежет на левую рабочую зону и на правую)
 - bash split_videos.sh        (Нарежет видео на 20180 сэмплов по 5 секунд каждый)

2. (Опционально. ann_train.txt и ann_test.txt уже в наличии, повторно запускать не требуется) python split_train_val.py - разбиваем на Тестовую и Обучающую выборки. Берем последний 1 час из five_hour_video и первые 2 часа из nine_hour_video в качестве Тестовой выборки. В результате получаем ann_train.txt и ann_test.txt.

3. bash train.sh
Запускает обучение модели на 2 GPU с размером видеопамяти от 48 GB. Для запуска на одной GPU или на более 2 нужно в train.sh изменить соответствующее значение. Если памяти менее 48 GB на карту, можно попробовать уменьшить batch_size в конфиге модели mvit_koblik_224_24_5sec.py

4. python inference_validation.py    (Предсказываем Тестовую выборку и показываем метрики и Confusion Matrix)