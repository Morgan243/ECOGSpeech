from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable
import uuid
import torch
from os.path import join as pjoin
from pathlib import Path
import os
import json

from typing import Optional

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

    def __post_init__(self):
        if self.save_model_path is None and self.result_dir is not None:
            self.save_model_path = pjoin(self.result_dir, 'models')
            print("Auto inferred save_model_path = " + self.save_model_path)

        if self.result_dir is not None:
            Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        if self.save_model_path is not None:
            Path(self.save_model_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ResultInputOptions(JsonSerializable):
    result_file: str = None
    model_base_path: Optional[str] = None

    def __post_init__(self):
        #logger.info((self.result_file, self.model_base_path))
        if self.result_file is not None and self.model_base_path is None:
            parent_dir, fname = os.path.split(self.result_file)
            #logger.info((parent_dir, fname))
            self.model_base_path = pjoin(parent_dir, 'models/')
            logger.info(f"Model base path inferred to be: {self.model_base_path}")


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
    def filter_dict_to_json_serializable(cls, d, recurse=True):
        o = dict()
        for k, v in d.items():
            if recurse and isinstance(v, dict):
                o[k] = cls.filter_dict_to_json_serializable(v, recurse=True)
            elif utils.is_jsonable(v):
                o[k] = v
        return o

    @classmethod
    def save_results(cls, model: torch.nn.Module,
                     name: str,
                     result_output: ResultOptions,
                     uid: str,
                     res_dict: dict,
                     filter_to_serializable: bool = True):
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
            o = cls.filter_dict_to_json_serializable(res_dict) if filter_to_serializable else res_dict
            with open(path, 'w') as f:
                json.dump(o, f)

        return res_dict
