from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable
import uuid
import torch
from os.path import join as pjoin
import os
import json

from typing import List, Optional, Type, ClassVar

from ecog_speech import utils
logger = utils.get_logger('experiments.base')


@dataclass
class TaskOptions(JsonSerializable):
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
class ResultInputOptions(JsonSerializable):
    result_file: str = None
    model_base_path: Optional[str] = None


@dataclass
class Experiment(JsonSerializable):
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

    @classmethod
    def save_results(cls, model: torch.nn.Module,
                     name: str,
                     result_output: ResultOptions,
                     uid: str,
                     res_dict: dict):
        if result_output.save_model_path is not None:
            p = result_output.save_model_path
            if os.path.isdir(p):
                p = pjoin(p, uid + '.torch')
            logger.info("Saving model to " + p)
            torch.save(model.cpu().state_dict(), p)
            res_dict['save_model_path'] = p

        if result_output.result_dir is not None:
            path = pjoin(result_output.result_dir, name)
            logger.info(path)
            res_dict['path'] = path
            with open(path, 'w') as f:
                json.dump(res_dict, f)

        return res_dict
