#import standard
#from ecog_speech.experiments.standard import make_model, make_datasets_and_loaders, default_option_kwargs
from ecog_speech.experiments import standard
from ecog_speech import utils
from dataclasses import dataclass, field
import json
from os.path import join as pjoin
from ecog_speech.experiments import base as bxp
from ecog_speech.models import base as bmp
from ecog_speech import datasets
from simple_parsing import ArgumentParser, choice, subgroups
from ecog_speech.models import base_transformers
import torch

logger = utils.get_logger('semi_supervised')


@dataclass
class SemisupervisedCodebookTaskOptions(bxp.TaskOptions):
    task_name: str = "semisupervised_codebook_training"
    ppl_weight: float = 100.


@dataclass
class SemiSupervisedExperiment(bxp.Experiment):
    model: bmp.ModelOptions = subgroups(
        {"cog2vec": base_transformers.Cog2VecOptions,},
        default=base_transformers.Cog2VecOptions()
    )

    dataset: datasets.DatasetOptions = subgroups(
        {"hvs": datasets.HarvardSentencesDatasetOptions,
         "nww": datasets.NorthwesternWordsDatasetOptions},
        default=datasets.HarvardSentencesDatasetOptions())

    task: bxp.TaskOptions = subgroups(
        {"semi_supervised": SemisupervisedCodebookTaskOptions},
        default=SemisupervisedCodebookTaskOptions())

    def run(self):
        dataset_map, dl_map, eval_dl_map = self.dataset.make_datasets_and_loaders()
        model, model_kws = self.model.make_model(dataset_map['train'])

        # Shake out any forward pass errors now by running example data through model
        with torch.no_grad():
            model(model.t_in)

        # Default lr reduce to False, only setup if at patience is set
        trainer_kws = dict(lr_adjust_on_cv_loss=False)
        if self.task.lr_adjust_patience is not None:
            print("Configuring LR scheduler for model")
            lr_schedule_kws = dict(patience=self.task.lr_adjust_patience, factor=self.task.lr_adjust_factor,
                                   verbose=True)
            trainer_kws.update(dict(lr_adjust_on_plateau_kws=lr_schedule_kws,
                                    lr_adjust_on_cv_loss=True,
                                    model_name_to_lr_adjust='model'))

        trainer = base_transformers.Cog2VecTrainer(model_map=dict(model=model), opt_map=dict(),
                                                   train_data_gen=dl_map['train'], cv_data_gen=eval_dl_map['cv'],
                                                   learning_rate=self.task.learning_rate,
                                                   early_stopping_patience=self.task.early_stopping_patience,
                                                   device=self.task.device,
                                                   **trainer_kws)

        # For some reason the codebook indices isn't always on the right device... so this seems to help force it over
        #trainer.model_map['model'].quantizer.codebook_indices = trainer.model_map['model'].quantizer.codebook_indices.to(trainer.device)

        #trainer.squeeze_first = False
        trainer.ppl_weight = self.task.ppl_weight

        losses = trainer.train(self.task.n_epochs)

        model.load_state_dict(trainer.get_best_state())

        #####
        # Produce predictions and score them
        model.eval()

        # outputs_map = trainer.generate_outputs(**eval_dl_map)
        eval_res_map = {k: trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}

        res_dict = self.create_result_dictionary(
            batch_losses=losses,
            train_selected_columns=dataset_map['train'].selected_columns,  # dataset_map['train'].selected_columns,
            best_model_epoch=trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(model),
            num_params=utils.number_of_model_params(model, trainable_only=False),
            model_kws=model_kws,
            **eval_res_map,
            **vars(self))

        uid = res_dict['uid']
        name = res_dict['name']

        if self.result_output.save_model_path is not None:
            import os
            p = self.result_output.save_model_path
            if os.path.isdir(p):
                p = os.path.join(p, uid + '.torch')
            logger.info("Saving model to " + p)
            torch.save(model.cpu().state_dict(), p)
            res_dict['save_model_path'] = p

        if self.result_output.result_dir is not None:
            path = pjoin(self.result_output.result_dir, name)
            logger.info(path)
            res_dict['path'] = path
            with open(path, 'w') as f:
                json.dump(res_dict, f)

        return trainer, eval_res_map


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SemiSupervisedExperiment, dest='semi_supervised')
    args = parser.parse_args()
    experiment: SemiSupervisedExperiment = args.semi_supervised
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
