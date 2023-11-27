from typing import Optional, List, Union

# Common
allow_snapshots: bool = False
audio_fs: int = 2000  # (in Hz)
wsize_sec: float = 1  # 5.0
wstep_sec: float = 2.5
audio_crop_sec: float = 2.0  # Crop start and stop of each audio recording, only during training (in total 2*crop_sec)

DEBUG: bool = False

# Self-supervised
ssl_epochs: int = 50
ssl_warmup_epochs: int = int(0.1 * ssl_epochs)
ssl_batch_size: int = 256  # Note that actual batch size is double for SSL because of dual augmentation
ssl_patience: int = 5
ssl_default_backbone: str = 'cnn'

ssl_default_temperature: float = 0.1  # 1.0
ssl_default_warmup_scaling: str = 'linear'
ssl_default_base_learning_rate: float = 0.1  # 0.3
ssl_default_offset: float = 0.0
ssl_default_augmentation1: Optional[Union[List[str], str]] = None  # ['cutofffilter_250_200', 'fliprandom']
ssl_default_augmentation2: Optional[Union[List[str], str]] = None  # ['uniformnoise_-0.01_0.01', 'zoom_2.0_0.5']

ssl_model_load_asset: bool = False

# Downstream: common
_epochs: int = 200
_batch_size: int = 64
_patience: int = int(0.1 * _epochs)
_adam_lr: float = 0.0001
ds_freeze: bool = True

# Downstream: common Resnet
# _epochs: int = 100
# _batch_size: int = 64
# _patience: int = 20

# Downstream: murmur
ds_murmur_epochs: int = _epochs
ds_murmur_batch_size: int = _batch_size
ds_murmur_patience: int = _patience
ds_murmur_adam_lr: float = _adam_lr
ds_murmur_class_weights: Optional[List[int]] = None  # [5, 3, 1]

# Downstream: outcome
ds_outcome_epochs: int = _epochs
ds_outcome_batch_size: int = _batch_size
ds_outcome_patience: int = _patience
ds_outcome_adam_lr: float = _adam_lr
ds_outcome_class_weights: Optional[List[int]] = None  # [5, 1]
