import standard
from ecog_speech.experiments.standard import make_model, make_datasets_and_loaders, default_option_kwargs
from ecog_speech import utils
from dataclasses import dataclass, field
from ecog_speech.experiments import base as bxp


logger = utils.get_logger('semi_supervised')


def run(options):
    from ecog_speech.models import base
    from ecog_speech.models import base_transformers
    import torch

    model_kws = dict(input_shape=(1, 256), feature_model=None, context_model=None, projection_model=None,
                     dropout=options.dropout, negatives_from_everywhere=True, feature_grad_mult=.1,
                     n_negatives=50, codebook_negatives=25, cross_sample_negatives=25,
                     mask_length=4, n_encoder_heads=options.n_encoder_heads, n_encoder_layers=options.n_encoder_layers,
                     quant_num_vars=options.quant_num_vars, quant_num_groups=options.quant_num_groups,
                     feature_extractor_layers=options.feature_extractor_layers)

    if options.model_name == 'cog2vec':
        model = base_transformers.CoG2Vec(**model_kws)
    else:
        raise ValueError(f"Don't understand model_name '{options.model_name}'")

    # Shake out any forward pass errors now by running example data through model
    with torch.no_grad():
        model(model.t_x)

    from ecog_speech import datasets

    # TODO: Need way to override defaults in options
    #options.pre_processing_pipeline = 'audio_gate_speaking_only'
    add_trfs = [datasets.SelectFromDim(dim=0, index='random', keep_dim=True)] if options.random_sensors_to_samples else None
    data_kws = {k: dict(flatten_sensors_to_samples=options.flatten_sensors_to_samples,
                        #additional_train_transforms=add_trfs, additional_eval_transforms=add_trfs
                )
                 for k in ['train_data_kws', 'cv_data_kws', 'test_data_kws']}
    dataset_map, dl_map, eval_dl_map = standard.make_datasets_and_loaders(options, dataset_cls=datasets.HarvardSentences,
                                                                          pre_processing_pipeline='audio_gate_speaking_only',
                                                                          additional_transforms=add_trfs,
                                                                          num_dl_workers=options.n_dl_workers,
                                                                          **data_kws)

    # Default lr reduce to False, only setup if at patience is set
    trainer_kws = dict(lr_adjust_on_cv_loss=False)
    if options.lr_adjust_patience is not None:
        print("Configuring LR scheduler for model")
        lr_schedule_kws = dict(patience=options.lr_adjust_patience, factor=options.lr_adjust_factor, verbose=True)
        trainer_kws.update(dict(lr_adjust_on_plateau_kws=lr_schedule_kws,
                                lr_adjust_on_cv_loss=True,
                                model_name_to_lr_adjust='model'))

    trainer = base_transformers.Cog2VecTrainer(model_map=dict(model=model), opt_map=dict(),
                                               train_data_gen=dl_map['train'], cv_data_gen=eval_dl_map['cv'],
                                               learning_rate=options.learning_rate,
                                               early_stopping_patience=options.early_stopping_patience,
                                               device=options.device,
                                               **trainer_kws)

    # For some reason the codebook indices isn't always on the right device... so this seems to help force it over
    trainer.model_map['model'].quantizer.codebook_indices = trainer.model_map['model'].quantizer.codebook_indices.to(trainer.device)

    trainer.squeeze_first = False
    trainer.ppl_weight = options.ppl_weight

    losses = trainer.train(options.n_epochs)

    model.load_state_dict(trainer.get_best_state())

    #####
    # Produce predictions and score them
    model.eval()
    #outputs_map = trainer.generate_outputs(**eval_dl_map)
    #clf_str_map = utils.make_classification_reports(outputs_map)

    #train_perf_map = utils.performance(outputs_map['train']['actuals'],
    #                                   outputs_map['train']['preds'] > 0.5)
    #cv_perf_map = utils.performance(outputs_map['cv']['actuals'],
    #                                outputs_map['cv']['preds'] > 0.5)
    #test_perf_map = utils.performance(outputs_map['test']['actuals'],
    #                                  outputs_map['test']['preds'] > 0.5)

    import uuid
    from datetime import datetime
    import time

    #####
    # Prep a results structure for saving - everything must be json serializable (no array objects)
    uid = str(uuid.uuid4())
    t = int(time.time())
    name = "%d_%s_TL.json" % (t, uid)
    res_dict = dict(  # path=path,
        name=name,
        datetime=str(datetime.now()), uid=uid,
        # batch_losses=list(losses),
        batch_losses=losses,
        train_selected_columns=dataset_map['train'].selected_columns,#dataset_map['train'].selected_columns,
        best_model_epoch=trainer.best_model_epoch,
        num_trainable_params=utils.number_of_model_params(model),
        num_params=utils.number_of_model_params(model, trainable_only=False),
        model_kws=model_kws,
        #clf_reports=clf_str_map,
        #**{'train_' + k: v for k, v in train_perf_map.items()},
        #**{'cv_' + k: v for k, v in cv_perf_map.items()},
        #**test_perf_map,
        # evaluation_perf_map=perf_maps,
        # **pretrain_res,
        # **perf_map,
        **vars(options))

    if options.save_model_path is not None:
        import os
        p = options.save_model_path
        if os.path.isdir(p):
            p = os.path.join(p, uid + '.torch')
        logger.info("Saving model to " + p)
        torch.save(model.cpu().state_dict(), p)
        res_dict['save_model_path'] = p


@dataclass
class SemiSupervisedOptions(bxp.DNNModelOptions, bxp.MultiSensorOptions):
    # ###
    model_name: str = 'cog2vec'      # Supported: {'cog2vec'}
    dataset: str = 'hvs'
    train_sets: str = 'UCSD-28'
    pre_processing_pipeline: str = "audio_gate_speaking_only"
    # ###

    feature_extractor_layers: str = '[(128, 7, 3)] + [(128, 3, 2)] * 2 + [(256, 3, 1)]'
    """String that evaluates to list of tuples describing 1-d convolution feature extractor 
        [(n-channels, kernel size, step size)..]"""

    quant_num_vars: int = 20
    """Number of variables in quantizer codebook"""
    quant_num_groups: int = 2
    """Number of groups in quantizer"""
    ppl_weight: float = 100
    """Weight of perplexity loss - use to encourage full use of codebook"""

    n_encoder_layers: int = 12
    n_encoder_heads: int = 8


all_model_hyperparam_names = [k for k, v in SemiSupervisedOptions.__annotations__.items()
                              if k not in ('train_sets', 'cv_sets', 'test_sets')]

if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SemiSupervisedOptions, dest='semi_supervised')
    args = parser.parse_args()
    main_options: SemiSupervisedOptions = args.semi_supervised
    main_options.random_sensors_to_samples = True
    run(main_options)
