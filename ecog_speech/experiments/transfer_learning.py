import uuid
import copy
import time
from datetime import datetime
from os.path import join as pjoin

import torch
from ecog_speech import utils
from ecog_speech.models import base
from typing import List, Optional
from tqdm.auto import tqdm
import numpy as np

import attr

from ecog_speech.models.sinc_ieeg import make_model
from ecog_speech.experiments import base as bxp
from ecog_speech.models import base_fine_tuners as base_ft
from ecog_speech import datasets
from dataclasses import dataclass
import json

logger = utils.get_logger(__name__)


#@dataclass
#class TransferLearningOptions(bmp.DNNModelOptions, bxp.MultiSensorOptions, Serializable):
#    model_name: Optional[str] = None
#    dataset: Optional[str] = 'hvs'
#    train_sets: Optional[str] = None
#
#    pretrained_result_input_path: Optional[str] = None
#    pretrained_result_model_base_path: Optional[str] = None
#
#    pre_train_sets: Optional[str] = None
#    pre_cv_sets: Optional[str] = None
#    pre_test_sets: Optional[str] = None
#
#    # 1d_linear, 2d_transformers
#    fine_tuning_method: str = '1d_linear'

from simple_parsing import subgroups
from ecog_speech import result_parsing

# Override to make the result parsing options optional in this script
@dataclass
class TransferLearningResultParsingOptions(result_parsing.ResultParsingOptions):
    result_file: Optional[str] = None
    print_results: Optional[bool] = False


@dataclass
class SpeechDetectionFineTuningTask(bxp.TaskOptions):
    task_name: str = "speech_classification_fine_tuning"
    dataset: datasets.DatasetOptions = datasets.DatasetOptions('hvs', train_sets='UCSD-22')
    method: str = '2d_transformers'


@dataclass
class RegionDetectionFineTuningTask(bxp.TaskOptions):
    task_name: str = "region_classification_fine_tuning"
    dataset: datasets.DatasetOptions = datasets.DatasetOptions('hvs', train_sets='AUTO-REMAINING',
                                                               pre_processing_pipeline='region_classification')
    method: str = '2d_linear'


@dataclass
class FineTuningExperiment(bxp.Experiment):
    pretrained_result_input: bxp.ResultInputOptions = None
    task: SpeechDetectionFineTuningTask = subgroups(
        {'speech_detection': SpeechDetectionFineTuningTask(),
         'region_detection': RegionDetectionFineTuningTask()},
        default=RegionDetectionFineTuningTask())
    # Don't need model options directly
    # TODO: make a fine tuning model options to capture at elast 'method' in task above
    #model: bmp.ModelOptions = subgroups(
    #    {
    #        'cog2vec': btf.Cog2VecOptions,
    #    },
    #    default=btf.Cog2VecOptions()
    #)

    @classmethod
    def make_pretrained_model(cls,
                              pretrained_result_input_path: str = None,
                              pretrained_result_model_base_path: str = None):
        #pretrained_result_model_base_path = pretrained_result_model_base_path if options is None else options.pretrained_result_model_base_path
        #pretrained_result_input_path = pretrained_result_input_path if options is None else options.pretrained_result_input_path

        assert_err = "pretrained_result_input_path must be populated as parameter or in options object"
        assert pretrained_result_input_path is not None, assert_err

        result_json = None
        from ecog_speech.result_parsing import load_model_from_results

        result_path = pretrained_result_input_path
        model_base_path = pretrained_result_model_base_path

        print(f"Loading pretrained model from results in {result_path} (base path = {model_base_path})")
        with open(result_path, 'r') as f:
            result_json = json.load(f)

        print(f"\tKEYS: {list(sorted(result_json.keys()))}")
        pretrained_model = load_model_from_results(result_json, base_model_path=model_base_path)

        return pretrained_model, result_json

    @property
    def pretrained_model(self):
        if not hasattr(self, '_pretrained_model'):
            self._pretrained_model, self._pretraining_results = self.make_pretrained_model(
                self.pretrained_result_input.result_file,
                self.pretrained_result_input.model_base_path)
        return self._pretrained_model, self._pretraining_results

    def make_fine_tuning_datasets_and_loaders(self, pretraining_sets=None):
        train_sets = None
        if self.task.dataset.train_sets == 'AUTO-REMAINING':
            pretraining_sets = self.pretrained_model[1]['dataset_options']['train_sets'] if pretraining_sets is None else pretraining_sets
            # Literally could have just .replace('~'), but instead wrote '*' special case for some set math in case it
            # gets more complicated...
            train_sets = list(set(datasets.HarvardSentences.make_tuples_from_sets_str('*'))
                              - set(datasets.HarvardSentences.make_tuples_from_sets_str(pretraining_sets)))
            logger.info(f"AUTO-REMAINING: pretrained on {pretraining_sets}, so fine tuning on {train_sets}")

        return self.task.dataset.make_datasets_and_loaders(train_p_tuples=train_sets)



    @classmethod
    def create_fine_tuning_model(cls, pretrained_model,
                                 n_pretrained_output_channels=None, n_pretrained_output_samples=None,
                                 fine_tuning_method='1d_linear',
                                 dataset: datasets.BaseDataset = None,
                                 fine_tuning_target_shape=None, n_pretrained_input_channels=None,
                                 n_pretrained_input_samples=256,
                                 freeze_pretrained_weights=True,
                                 classifier_head: torch.nn.Module = None):
        from ecog_speech.models import base
        n_pretrained_output_channels = pretrained_model.C if n_pretrained_output_channels is None else n_pretrained_output_samples
        n_pretrained_output_samples = pretrained_model.T if n_pretrained_output_samples is None else n_pretrained_output_samples

        n_pretrained_input_channels = n_pretrained_input_channels if dataset is None else len(dataset.selected_columns)
        # n_pretrained_input_samples = n_pretrained_input_samples if dataset is None else dataset.get_target_shape()
        fine_tuning_target_shape = dataset.get_target_shape() if fine_tuning_target_shape is None else fine_tuning_target_shape

        m = copy.deepcopy(pretrained_model)
        m.quantizer.codebook_indices = None
        if freeze_pretrained_weights:
            for param in m.parameters():
                param.requires_grad = False

        if fine_tuning_method == '1d_linear':
            # Very simple linear classifier head by default
            if classifier_head is None:
                h_size = 32
                classifier_head = torch.nn.Sequential(*[
                    base.Reshape((n_pretrained_output_channels * n_pretrained_output_samples,)),
                    torch.nn.Linear(n_pretrained_output_channels * n_pretrained_output_samples, h_size),
                    # torch.nn.BatchNorm1d(h_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(h_size, h_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(h_size, fine_tuning_target_shape),
                    # torch.nn.Sigmoid()
                ])
                # nn.init.xavier_uniform_(classifier_head.weight)

            ft_model = base_ft.FineTuner(pre_trained_model=m, output_model=classifier_head,
                                         pre_trained_model_forward_kws=dict(features_only=True, mask=False),
                                         pre_trained_model_output_key='x',
                                         freeze_pre_train_weights=freeze_pretrained_weights)

        elif '2d_' in fine_tuning_method :
            if dataset is None:
                raise ValueError(f"A dataset is required for '2d_*' methods in order to see num sensors")
            from ecog_speech.models import base_transformers

            hidden_enc = 'transformer' if fine_tuning_method == '2d_transformers' else 'linear'
            ft_model = base_transformers.MultiChannelCog2Vec((n_pretrained_input_channels, n_pretrained_input_samples),
                                                             pretrained_model, outputs=dataset.get_target_shape(),
                                                             hidden_encoder=hidden_enc)
        else:
            raise ValueError(f"Unknown ft_method '{fine_tuning_method}'")

        return ft_model

    def run(self):
        # Pretrained model already prepared, parse from its results output
        pretrained_model, pretraining_results = self.pretrained_model
        #pretrained_model, pretraining_results = self.make_pretrained_model(self.pretrained_result_input.result_file,
        #                                                                   self.pretrained_result_input.model_base_path)
        #train_sets = None
        #if self.task.dataset.train_sets == 'AUTO-REMAINING':
        #    # Literally could have just .replace('~'), but instead wrote '*' special case for some set math in case it
        #    # gets more complicated...
        #    pretrained_set = pretraining_results['dataset_options']['train_sets']
        #    train_sets = list(set(datasets.HarvardSentences.make_tuples_from_sets_str('*'))
        #                      - set(datasets.HarvardSentences.make_tuples_from_sets_str(pretrained_set)))
        #    logger.info(f"AUTO-REMAINING: pretrained on {pretrained_set}, so fine tuning on {train_sets}")

        #dataset_map, dl_map, eval_dl_map = self.task.dataset.make_datasets_and_loaders(train_p_tuples=train_sets)

        dataset_map, dl_map, eval_dl_map = self.make_fine_tuning_datasets_and_loaders()

        # Capture configurable kws separately, so they can be easily saved in the results at the end
        fine_tune_model_kws = dict(fine_tuning_method=self.task.method)
        fine_tune_model = self.create_fine_tuning_model(pretrained_model, dataset=dataset_map['train'],
                                                        **fine_tune_model_kws)

        # Decide how to setup the loss depending on the dataset and task - TODO: should clean this up?
        dset_name = self.task.dataset.dataset_name
        squeeze_target = False
        if dset_name == 'hvsmfc':
            criterion = torch.nn.MSELoss()
            target_key = 'target'
        elif self.task.task_name == 'speech_classification_fine_tuning':
            pos_weight = torch.FloatTensor([0.5]).to(self.task.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            target_key = 'target_arr'
        elif self.task.task_name == 'region_classification_fine_tuning':
            #criterion = torch.nn.BCEWithLogitsLoss()
            criterion = torch.nn.CrossEntropyLoss()
            target_key = 'target_arr'
            squeeze_target = True
        else:
            raise ValueError(f"Don't understand {dset_name} and {self.task.task_name}")

        logger.info(f"Criterion for {self.task.task_name}: {criterion} on {target_key}")

        trainer_cls = TLTrainer
        ft_trainer = trainer_cls(model_map=dict(model=fine_tune_model), opt_map=dict(),
                                  train_data_gen=dl_map['train'],
                                  cv_data_gen=eval_dl_map.get('cv'),
                                  input_key='signal_arr',
                                  learning_rate=self.task.learning_rate,
                                  early_stopping_patience=self.task.early_stopping_patience,
                                  target_key=target_key,
                                  criterion=criterion,
                                  device=self.task.device,
                                 squeeze_target=squeeze_target
                                  )

        #with torch.no_grad():
        #    _outputs_map = ft_trainer.generate_outputs(train=eval_dl_map['train'])
        #    _clf_str_map = utils.make_classification_reports(_outputs_map)

        ft_results = ft_trainer.train(self.task.n_epochs)

        fine_tune_model.load_state_dict(ft_trainer.get_best_state())

        fine_tune_model.eval()
        outputs_map = ft_trainer.generate_outputs(**eval_dl_map)

        performance_map = dict()
        if dset_name == 'hvsmfc':
            pass
        elif dset_name == 'hvs':
            target_shape = dataset_map['train'].get_target_shape()
            #eval_res_map = {k: ft_trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}
            kws = dict(threshold=(0.5 if target_shape == 1 else None))
            clf_str_map = utils.make_classification_reports(outputs_map, **kws)
            if target_shape == 1:
                performance_map = {part_name: utils.performance(outputs_d['actuals'],
                                                                outputs_d['preds'] > 0.5)
                                   for part_name, outputs_d in outputs_map.items()}
            else:
                performance_map = {part_name: utils.multiclass_performance(outputs_d['actuals'],
                                                                           outputs_d['preds'].argmax(1))
                                   for part_name, outputs_d in outputs_map.items()}

        #####
        # Prep a results structure for saving - everything must be json serializable (no array objects)
        res_dict = self.create_result_dictionary(
            #model_name=self.model.model_name,
            batch_losses=ft_results,
            train_selected_columns=dataset_map['train'].selected_columns,  # dataset_map['train'].selected_columns,
            #test_selected_flat_indices=dataset_map['test'].selected_flat_indices,
            #selected_flat_indices={k: d.selected_flat_indices for k, d in dataset_map.items()},
            selected_flat_indices={k: d.selected_levels_df.to_json() for k, d in dataset_map.items()},
            best_model_epoch=ft_trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(fine_tune_model),
            num_params=utils.number_of_model_params(fine_tune_model, trainable_only=False),
            model_kws=fine_tune_model_kws,
            **performance_map,
            #**eval_res_map,
            pretrained_result_input=vars(self.pretrained_result_input),
            task_options=vars(self.task),
            #dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output)
        )
        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(fine_tune_model, name, result_output=self.result_output, uid=uid, res_dict=res_dict)

        return ft_trainer, performance_map


        #uid = str(uuid.uuid4())
        #t = int(time.time())
        #name = "%d_%s_TL.json" % (t, uid)
        #res_dict = dict(  # path=path,
        #    name=name,
        #    datetime=str(datetime.now()), uid=uid,
        #    pretraining_results=pretraining_results,
        #    # batch_losses=list(losses),
        #    batch_losses=ft_results,
        #    train_selected_columns=dataset_map['train'].selected_columns,#dataset_map['train'].selected_columns,
        #    best_model_epoch=ft_trainer.best_model_epoch,
        #    num_trainable_params=utils.number_of_model_params(fine_tune_model),
        #    num_params=utils.number_of_model_params(fine_tune_model, trainable_only=False),
        #    fine_tune_model_kws=fine_tune_model_kws,
        #    options=self.dumps_json(),
        #    #model_kws=model_kws,
        #    #**eval_res_map,
        #    #clf_reports=clf_str_map,
        #    #**{'train_' + k: v for k, v in train_perf_map.items()},
        #    #**{'cv_' + k: v for k, v in cv_perf_map.items()},
        #    #**test_perf_map,
        #    # evaluation_perf_map=perf_maps,
        #    # **pretrain_res,
        #    # **perf_map,
        #    **vars(self))

        #if self.result_output.save_model_path is not None:
        #    import os
        #    p = self.result_output.save_model_path
        #    if os.path.isdir(p):
        #        p = os.path.join(p, uid + '.torch')
        #    logger.info("Saving model to " + p)
        #    torch.save(fine_tune_model.cpu().state_dict(), p)
        #    res_dict['save_model_path'] = p

        #if self.result_output.result_dir is not None:
        #    path = pjoin(self.result_output.result_dir, name)
        #    logger.info(path)
        #    res_dict['path'] = path
        #    with open(path, 'w') as f:
        #        json.dump(res_dict, f)


@attr.s
class TLTrainer(base.Trainer):
    input_key = attr.ib('signal_arr')
    squeeze_target = attr.ib(False)
    squeeze_first = True

    def loss(self, model_output_d, input_d, as_tensor=True):
        crit_loss = self.criterion(model_output_d,
                                   input_d[self.target_key].squeeze() if self.squeeze_target
                                   else input_d[self.target_key])
        return crit_loss

    def _eval(self, epoch_i, dataloader, model_key='model'):
        """
        trainer's internal method for evaluating losses,
        snapshotting best models and printing results to screen
        """
        model = self.model_map[model_key].eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    input_d = {k: v.to(self.device) for k, v in _x.items()}
                    #input_arr = input_d[self.input_key]
                    #actual_arr = input_d[self.target_key]
                    m_output = model(input_d)

                    #actuals = input_d[self.target_key]

                    #loss = self.criterion(preds, actuals)
                    loss = self.loss(m_output, input_d)

                    loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                reg_l = 0.
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))

                overall_loss = (mean_loss + reg_l)

                if overall_loss < self.best_cv:

                    self.best_model_state = self.copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return dict(primary_loss=overall_loss, cv_losses=loss_l)

    def train_inner_step(self, epoch_i, data_batch):
        """
        Core training method - gradient descent - provided the epoch number and a batch of data and
        must return a dictionary of losses.
        """
        res_d = dict()

        model = self.model_map['model'].to(self.device)
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        input_d = {k: v.to(self.device) for k, v in data_batch.items()}
        input_arr = input_d[self.input_key]
        actual_arr = input_d[self.target_key]
        #m_output = model(input_arr)
        m_output = model(input_d)

        #crit_loss = self.criterion(m_output, actual_arr)
        #crit_loss = self.loss(m_output, actual_arr)
        crit_loss = self.loss(m_output, input_d)
        res_d['crit_loss'] = crit_loss.detach().cpu().item()

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            res_d['bwreg'] = reg_l.detach().cpu().item()
        else:
            reg_l = 0

        loss = crit_loss + reg_l
        res_d['total_loss'] = loss.detach().cpu().item()
        loss.backward()
        optim.step()
        model = model.eval()
        return res_d


    def generate_outputs_from_model_inner_step(self, model, data_batch, criterion=None,
                                               input_key='signal_arr', target_key='text_arr', device=None,
                                               ):
        #X = data_batch[input_key].to(device)

        #if self.squeeze_first:
        #    X = X.squeeze()

        with torch.no_grad():
            model.eval()
            model.to(device)
            input_d = {k: v.to(self.device) for k, v in data_batch.items()}
            preds = model(input_d)

            #loss_v = self.loss(preds, input_d)
        #score_d = self.score_training(m_d, as_tensor=True)
        eval_d = dict(preds=preds, actuals=input_d[self.target_key], #loss=torch.tensor(loss_v)
                      )#dict(**score_d, **loss_d)

        return eval_d


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="ASPEN+MHRG Transfer Learning experiments")
    parser.add_arguments(FineTuningExperiment, dest='transfer_learning')
    #parser.add_arguments(TransferLearningOptions, dest='transfer_learning')
    #parser.add_arguments(TransferLearningResultParsingOptions, dest='tl_result_parsing')
    args = parser.parse_args()
    tl: FineTuningExperiment = args.transfer_learning
    tl.run()
