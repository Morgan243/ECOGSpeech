from ecog_speech import utils
from dataclasses import dataclass, field
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
        {"cog2vec": base_transformers.Cog2VecOptions, 'dummy': base_transformers.Cog2VecOptions},
        default=base_transformers.Cog2VecOptions()
    )

    dataset: datasets.DatasetOptions = subgroups(
        {"hvs": datasets.HarvardSentencesDatasetOptions,
         "nww": datasets.NorthwesternWordsDatasetOptions},
        default=datasets.HarvardSentencesDatasetOptions())

    task: bxp.TaskOptions = subgroups(
        {"semi_supervised": SemisupervisedCodebookTaskOptions, "dummy": SemisupervisedCodebookTaskOptions},
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
            model_name=self.model.model_name,
            batch_losses=losses,
            train_selected_columns=dataset_map['train'].selected_columns,  # dataset_map['train'].selected_columns,
            best_model_epoch=trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(model),
            num_params=utils.number_of_model_params(model, trainable_only=False),
            model_kws=model_kws,
            **eval_res_map,
            model_options=vars(self.model),
            task_options=vars(self.dataset),
            dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output)
        )

        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(model, name, result_output=self.result_output, uid=uid, res_dict=res_dict)

        return trainer, eval_res_map


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SemiSupervisedExperiment, dest='semi_supervised')
    args = parser.parse_args()
    experiment: SemiSupervisedExperiment = args.semi_supervised
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
