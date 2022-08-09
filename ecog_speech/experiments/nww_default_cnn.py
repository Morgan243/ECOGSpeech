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
    task_name: str = "default_cnn"
    ppl_weight: float = 100.


@dataclass
class DefaultCNNExperiment(bxp.Experiment):
    model: bmp.ModelOptions = subgroups(
        {"cog2vec": base_transformers.Cog2VecOptions,
         'default_cnn': bmp.BaseCNNModelOptions},
        default=bmp.BaseCNNModelOptions()
    )

    dataset: datasets.DatasetOptions = subgroups(
        {

            "hvs": datasets.HarvardSentencesDatasetOptions(pre_processing_pipeline='random_sample'),
             # Not actually tested
            "nww-d": datasets.NWWDefaultDatasetOptions(pre_processing_pipeline='speech_activity_classification'),
            "nww-c": datasets.NWWChangDatasetOptions(pre_processing_pipeline='speech_activity_classification'),
            "nww-h": datasets.NWWHerffDatasetOptions(pre_processing_pipeline='speech_activity_classification')
        },
        default=datasets.NWWDefaultDatasetOptions(pre_processing_pipeline='speech_activity_classification'))

    task: bxp.TaskOptions = subgroups(
        {"default_cnn": SemisupervisedCodebookTaskOptions,
         "dummy": SemisupervisedCodebookTaskOptions},
        default=SemisupervisedCodebookTaskOptions())

    def run(self):
        # Reduce default test size for sklearn train/test split from 0.25 to 0.2
        dataset_map, dl_map, eval_dl_map = self.dataset.make_datasets_and_loaders()
        # c_dataset = datasets.NWWChangDatasetOptions(pre_processing_pipeline='speech_activity_classification')
        # h_datatset = datasets.NWWHerffDatasetOptions(pre_processing_pipeline='speech_activity_classification')
        # dataset_map_c, dl_map_c, eval_dl_map_c = c_dataset.make_datasets_and_loaders()
        # dataset_map_h, dl_map_h, eval_dl_map_h = h_datatset.make_datasets_and_loaders()

        model, model_kws = self.model.make_model(dataset_map['train'])

        # Shake out any forward pass errors now by running example data through model - the model has a small random
        # tensor t_in that can be pass in

        with torch.no_grad():
            model(model.t_in)

        # Default lr reduce to False, only setup if at patience is set
        trainer_kws = dict(lr_adjust_on_cv_loss=False)
        if self.task.lr_adjust_patience is not None:
            logger.info(f"Configuring LR scheduler for model: patience={self.task.lr_adjust_patience}")
            lr_schedule_kws = dict(patience=self.task.lr_adjust_patience, factor=self.task.lr_adjust_factor)
            trainer_kws.update(dict(lr_adjust_on_plateau_kws=lr_schedule_kws,
                                    lr_adjust_on_cv_loss=True,
                                    # Needs to match a model name in the model map passed to trainer below
                                    model_name_to_lr_adjust='model'))

        trainer = bmp.Trainer(model_map=dict(model=model), opt_map=dict(),
                                                   train_data_gen=dl_map['train'], cv_data_gen=eval_dl_map['cv'],
                                                   learning_rate=self.task.learning_rate,
                                                   early_stopping_patience=self.task.early_stopping_patience,
                                                   device=self.task.device,
                                                   **trainer_kws)

        # For some reason the codebook indices isn't always on the right device... so this seems to help force it over
        #trainer.model_map['model'].quantizer.codebook_indices = trainer.model_map['model'].quantizer.codebook_indices.to(trainer.device)

        #trainer.squeeze_first = False

        #####
        # Train
        losses = trainer.train(self.task.n_epochs)

        # reload the best model from memory
        model.load_state_dict(trainer.get_best_state())

        #####
        # Produce predictions and score them
        model.eval()

        # Produce mapping from dataloader names (train/cv/test) to dataframe of batch eval losses
        # This is different from a classification or other model - don't have a way to easily produce stats aggregated
        # across the whole dataset
        eval_res_map = {k: trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}

        # Create the dictionary that will be json serialized as the results
        res_dict = self.create_result_dictionary(
            model_name=self.model.model_name,
            epoch_outputs=losses,
            train_selected_columns=dataset_map['train'].selected_columns,
            selected_flat_indices={k: d.selected_levels_df.to_json() for k, d in dataset_map.items()},
            best_model_epoch=trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(model),
            num_params=utils.number_of_model_params(model, trainable_only=False),
            model_kws=model_kws,
            **eval_res_map,
            model_options=vars(self.model),
            task_options=vars(self.task),
            dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output)
        )

        # Grab the generated uid and name to use them as file names
        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(model, result_file_name=name, result_output=self.result_output,
                          model_file_name=uid, res_dict=res_dict)

        return trainer, eval_res_map


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(DefaultCNNExperiment, dest='default_cnn')
    args = parser.parse_args()
    experiment: DefaultCNNExperiment = args.default_cnn
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
