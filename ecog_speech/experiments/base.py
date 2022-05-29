from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable
import uuid

from typing import List, Optional, Type, ClassVar

from ecog_speech.models.base import ModelOptions
from ecog_speech.datasets import DatasetOptions


@dataclass
class TaskOptions(JsonSerializable):
    task_name: str = None
    dataset: DatasetOptions = None

    n_epochs: int = 100

    learning_rate: float = 0.001
    lr_adjust_patience: Optional[float] = None
    lr_adjust_factor: float = 0.1

    early_stopping_patience: Optional[int] = None

    device: Optional[str] = None


@dataclass
class ResultOptions(JsonSerializable):
    result_dir: Optional[str] = None
    save_model_path: Optional[str] = None


@dataclass
class Experiment(JsonSerializable):
    model: ModelOptions = None
    task: TaskOptions = None
    result_output: ResultOptions = field(default_factory=ResultOptions)
    tag: Optional[str] = None

    @classmethod
    def create_result_dictionary(cls, **kws):
        from datetime import datetime
        dt = datetime.now()
        dt_str = dt.strftime('%Y%m%d_%H%M')
        uid = str(uuid.uuid4())
        name = "%s_%s.json" % (dt_str, uid)
        res_dict = dict(  # path=path,
            name=name,
            datetime=str(dt), uid=uid,
            **kws
        )
        return res_dict


# -----
@dataclass
class BaseExperimentOptions(JsonSerializable):
    model_name: str = None
    dataset: str = None
    pre_processing_pipeline: str = 'default'

    train_sets: str = None
    cv_sets: Optional[str] = None
    test_sets: Optional[str] = None

    data_subset: str = 'Data'
    output_key: str = 'signal_arr'
    extra_output_keys: Optional[str] = None

    result_dir: Optional[str] = None
    save_model_path: Optional[str] = None
    device: Optional[str] = None
    tag: Optional[str] = None

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
