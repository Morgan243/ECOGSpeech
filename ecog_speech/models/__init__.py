from ecog_speech.models import sinc_ieeg, base_transformers


def make_model(options=None, nww=None, model_name=None, model_kws=None, print_details=True):

    if model_name == 'cog2vec':
        m = base_transformers.CoG2Vec(**model_kws)
        m_kws = model_kws
    else:
        m, m_kws = sinc_ieeg.make_model(options=options, nww=nww,
                                        model_name=model_name, model_kws=model_kws,
                                        print_details=print_details)
    return m, m_kws
