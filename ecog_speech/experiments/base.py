from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable

from typing import List, Optional

@dataclass
class BaseExperimentOptions(JsonSerializable):
    model_name: str = None
    dataset: str = None
    pre_processing_pipeline: str = 'default'

    train_sets: str = None
    cv_sets: Optional[str] = None
    test_sets: Optional[str] = None

    data_subset: str = 'Data'

    result_dir: Optional[str] = None
    save_model_path: Optional[str] = None
    device: Optional[str] = None

    @classmethod
    def get_all_model_hyperparam_names(cls):
        return [k for k, v in cls.__annotations__.items()
                 if k not in ('train_sets', 'cv_sets', 'test_sets')]


@dataclass
class TrainingExperimentOptions(BaseExperimentOptions):
    n_epochs: int = 100
    batch_size: int = 256
    batches_per_epoch: Optional[int] = None
    """If set, only does this many batches in an epoch - otherwise, will do enough batches to equal dataset size"""

    learning_rate: float = 0.001
    lr_adjust_patience: Optional[float] = None
    lr_adjust_factor: float = 0.1

    early_stopping_patience: Optional[int] = None

@dataclass
class MultiSensorOptions:
    flatten_sensors_to_samples: bool = False
    """Sensors will bre broken up into sensors - inputs beceome (1, N-timesteps) samples (before batching)"""
    random_sensors_to_samples: bool = False

@dataclass
class DNNModelOptions(TrainingExperimentOptions):
    activation_class: str = 'PReLU'
    dropout: float = 0.
    batchnorm: bool = False

    n_dl_workers: int = 4
    n_dl_eval_workers: int = 6

@dataclass
class FromResultOptions:
    result_input_path: Optional[str] = None
    result_model_base_path: Optional[str] = None
