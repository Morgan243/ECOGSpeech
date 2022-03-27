import standard
from ecog_speech.experiments.standard import make_model, make_datasets_and_loaders, default_option_kwargs
from ecog_speech import utils

logger = utils.get_logger('semi_supervised')

def run(options):
    from ecog_speech.models import base_transformers
    import torch

    model_kws = dict(input_shape=(1, 256), feature_model=None, context_model=None, projection_model=None,
                        negatives_from_everywhere=True, feature_grad_mult=.1,
                        n_negatives=50, codebook_negatives=25, cross_sample_negatives=25,
                        mask_length=4, n_encoder_heads=options.n_encoder_heads, n_encoder_layers=options.n_encoder_layers,
                     quant_num_vars=options.quant_num_vars,
                     quant_num_groups=options.quant_num_groups)

    model = base_transformers.CoG2Vec(**model_kws)

    # Shake out any forward pass errors now by running example data through model
    with torch.no_grad():
        model(model.t_x)

    from ecog_speech import datasets

    # TODO: Need way to override defaults in options
    #options.pre_processing_pipeline = 'audio_gate_speaking_only'
    dataset_map, dl_map, eval_dl_map = standard.make_datasets_and_loaders(options, dataset_cls=datasets.HarvardSentences,
                                                                          pre_processing_pipeline='audio_gate_speaking_only')

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


semisuper_options = [
    dict(dest='--ppl-weight', default=10, type=float),
    dict(dest='--quant-num-vars', default=10, type=int),
    dict(dest='--quant-num-groups', default=2, type=int),

    dict(dest='--n-encoder-layers', default=12, type=int),
    dict(dest='--n-encoder-heads', default=8, type=int),
]

ss_option_kwargs = default_option_kwargs + semisuper_options


all_model_hyperparam_names = [d['dest'].replace('--', '').replace('-', '_')
                              for d in semisuper_options
                              if d['dest'] not in ('--train-sets', '--cv-sets', '--test-sets')]

if __name__ == """__main__""":
    parser = utils.build_argparse(ss_option_kwargs,
                                  description="ASPEN+MHRG Semi-supervise learning experiments")
    m_options = parser.parse_args()
    run(m_options)
