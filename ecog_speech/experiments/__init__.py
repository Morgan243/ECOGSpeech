from ecog_speech.experiments import standard, transfer_learning
all_model_hyperparam_names = standard.all_model_hyperparam_names + transfer_learning.all_model_hyperparam_names

def make_model(options=None, nww=None, model_name=None, model_kws=None, print_details=True):
    #model, _ = experiments.make_model(model_name=results['model_name'], model_kws=model_kws)
    from ecog_speech import datasets
    if model_name == 'cog2vec':
        from ecog_speech.models.base_transformers import CoG2Vec
        m = CoG2Vec(**model_kws)
        m_kws = model_kws
    else:
        m, m_kws = standard.make_model(options=options, nww=nww,
                                         model_name=model_name, model_kws=model_kws,
                                         print_details=print_details)
    return m, m_kws
