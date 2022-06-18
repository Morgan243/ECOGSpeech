from sklearn.metrics import classification_report
from collections import namedtuple
import logging
import sys
import json


# https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_logger(logname='ecog_speech', console_level=logging.DEBUG, file_level=logging.DEBUG,
               format_string='%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
               #format_string='%(asctime)s - %(module)s.%(funcName)s - %(levelname)s - %(message)s',
               output_file=None):
    logger = logging.getLogger(logname)
    if logger.hasHandlers():
        return logging.getLogger(logname)
    else:
        pass
        #print("MAKING NEW LOGGER: " + logname)
    #logger = < create_my_logger > if not logging.getLogger().hasHandlers() else logging.getLogger()
    # create logger with 'spam_application'
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format_string)

    # create file handler which logs even debug messages
    if output_file is not None:
        fh = logging.FileHandler(output_file)
        fh.setLevel(file_level)
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    # create formatter and add it to the handlers
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    return logger


def with_logger(cls=None, prefix_name=None):
    def _make_cls(cls):
        n = __name__ if prefix_name is None else prefix_name
        cls.logger = get_logger(n + '.' + cls.__name__)
        return cls

    cls = _make_cls if cls is None else _make_cls(cls)

    return cls


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


def performance(y, preds):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    return dict(f1=f1_score(y, preds),
                accuracy=accuracy_score(y, preds),
                precision=precision_score(y, preds),
                recall=recall_score(y, preds),
                )


def make_classification_reports(output_map, pretty_print=True):
    out_d = dict()
    for dname, o_map in output_map.items():
        report_str = classification_report(o_map['actuals'], (o_map['preds'] > 0.5))
        if pretty_print:
            print("-"*10 + str(dname) + "-"*10)
            print(report_str)
        out_d[dname] = report_str
    return out_d


def build_argparse(default_option_kwargs, description=''):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    for _kwargs in default_option_kwargs:
        first_arg = _kwargs.pop('dest')
        parser.add_argument(first_arg, **_kwargs)

    return parser