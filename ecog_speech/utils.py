from collections import namedtuple

def print_sequential_arch(m, t_x):
    for i in range(0, len(m)):
        print(m[i])
        l_preds = m[:i + 1](t_x)
        print(l_preds.shape)
        print("----")

def number_of_model_params(m, trainable_only=True):
    p_cnt = sum(p.numel() for p in m.parameters()
                if (p.requires_grad and trainable_only) or not trainable_only)
    return p_cnt

def build_default_options(default_option_kwargs, **overrides):
    opt_keys = [d['dest'].replace('-', '_')[2:]
                for d in default_option_kwargs]
    Options = namedtuple('Options', opt_keys)
    return Options(*[d['default'] if o not in overrides else overrides[o]
                     for o, d in zip(opt_keys, default_option_kwargs)])

def build_argparse(default_option_kwargs, description=''):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    for _kwargs in default_option_kwargs:
        first_arg = _kwargs.pop('dest')
        parser.add_argument(first_arg, **_kwargs)

    return parser