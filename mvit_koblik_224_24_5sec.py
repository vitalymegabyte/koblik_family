# Model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MViT',
        arch='small',
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth',
            prefix='backbone.')),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type='MViTHead',
        in_channels=768,
        num_classes=3,
        label_smooth_eps=0.2,
        dropout_ratio=0.65,
        average_clips='prob'))

# Logging settings
default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False

# Specify dataset paths
dataset_type = 'VideoDataset'
data_root = './'
data_root_val = './'
ann_file_train = './ann_train.txt'
ann_file_val = './ann_test.txt'
ann_file_test = './ann_test.txt'


train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),

    dict(type='UniformSample', clip_len=24, test_mode=False),
    dict(type='DecordDecode'),

    dict(type='RandomRescale', scale_range=(160, 288)),
    dict(type='RandomResizedCrop', area_range=(0.5, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),

    dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', flip_ratio=0.5, direction='vertical'),

    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=8,
        num_layers=4,
        prob=0.8),

    dict(type='RandomErasing', erase_prob=0.3, mode='rand'),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', io_backend='disk'),

    dict(type='UniformSample', clip_len=24, test_mode=True),
    dict(type='DecordDecode'),

    dict(type='Resize', scale=(224, 224), keep_ratio=False),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),

    dict(type='UniformSample', clip_len=24, test_mode=True),
    dict(type='DecordDecode'),

    dict(type='Resize', scale=(224, 224), keep_ratio=False),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))


# Training settigns
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=4, val_begin=1, val_interval=1) #65
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 6e-5 # 0.0016 # 1e-4
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.1,
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=2,
        eta_min=base_lr / 3,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True)
]

auto_scale_lr = dict(enable=False, base_batch_size=64)

dist_params = dict(backend='nccl')
launcher = 'pytorch'
work_dir = 'work_dirs/mvit_koblik_224_32_5sec'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)

load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth'
checkpoint = load_from