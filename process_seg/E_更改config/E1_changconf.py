import os
os.chdir('E:\Square\mmsegmentation')
print(os.getcwd())

from mmengine import Config
cfg = Config.fromfile('./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('./configs/_base_/datasets/SquareDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg)

#背景，广场，空地
NUM_CLASS=3
cfg.crop_size=(512, 512)

cfg.model.data_preprocessor.size=cfg.crop_size

cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

cfg.train_dataloader.batch_size=4

#保存修改后的文件
cfg.work_dir = './work_dirs/MyDataset-DeepLabV3plus'

cfg.train_cfg.max_iters=20000
cfg.train_cfg.val_interval=500
cfg.default_hooks.logger.interval=500
cfg.default_hooks.checkpoint.interval=2500
cfg.default_hooks.checkpoint.max_keep_ckpts=1
cfg.default_hooks.checkpoint.save_best='mIoU'

cfg['randomness']=dict(seed=0)

print(cfg.pretty_text)
#保存最终的配置文件
cfg.dump('MyConfigs/SquareDataset_DeepLabV3plus.py')