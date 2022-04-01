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

    save_model_path: Optional[str] = None
    device: Optional[str] = None

@dataclass
class TrainingExperimentOptions(BaseExperimentOptions):
    n_epochs: int = 100
    batch_size: int = 256

    learning_rate: float = 0.001
    lr_adjust_patience: Optional[float] = None
    lr_adjust_factor: float = 0.1

    early_stopping_patience: Optional[int] = None

@dataclass
class MultiSensorOptions:
    flatten_sensors_to_samples: bool = False

@dataclass
class DNNModelOptions(TrainingExperimentOptions):
    activation_class: str = 'PReLU'
    dropout: float = 0.
    batchnorm: bool = False
