import torch
import json
import os

from ecog_speech.models import base, sinc_ieeg
from ecog_speech import utils
from lslkit.components import processor


def deploy_to_brainflow(options, result_file, brainflow_options):
    pass


def deploy_to_lsl(options, result_file, stream_type, max_buflen=2048):
    result_base_path, result_filename = os.path.split(result_file)
    result_id = result_filename.split('.')[0]

    # Load results to get the file name of the model
    results = json.load(open(result_file))

    model_kws = results['model_kws']
    base_model_path = options.base_model_path
    if base_model_path is None:
        base_model_path = os.path.join(result_base_path, 'models')
        print("Base model path not give - assuming path '%s'" % base_model_path)

    model_filename = os.path.split(results['save_model_path'])[-1]
    model_path = os.path.join(base_model_path, model_filename)
    print("Loading model located at: " + str(model_path))

    if results['model_name'] == 'base-sn':
        model = base.BaseMultiSincNN(**model_kws)
    elif results['model_name'] == 'tnorm-base-sn':
        model = sinc_ieeg.TimeNormBaseMultiSincNN(**model_kws)
    elif results['model_name'] == 'base-cnn':
        model = base.BaseCNN(**model_kws)
    else:
        raise ValueError(f"Unrecognized model_name: {results['model_name']} in {result_file})")

    with open(model_path, 'rb') as f:
        model_state = torch.load(f)

    model.load_state_dict(model_state)
    model.to(options.device)

    def run_on_df(x_df):
        x_arr = torch.from_numpy(x_df.values).float().T.unsqueeze(0)
        with torch.no_grad():
            y_arr = model(x_arr)
        # return pd.Series(y_arr.cpu().detach().numpy(), index=x_df.index)
        return y_arr.squeeze().item()

    #### LSL ####
    proc = processor.ProcessStream.from_resolve(run_on_df, stream_type=stream_type,
                                                max_buflen=max_buflen)
    proc.begin(required_size=model_kws['window_size'])

    # TODO: Save outputs


default_option_kwargs = [
    dict(dest="--result-file", default=None, type=str, required=True),
    dict(dest="--base-model-path", default=None, type=str),
   # dict(dest='--eval-sets', default=None, type=str,
   #      help="Dataset to run the loaded model against - use train/cv/test for the data used to build the model"
   #           "or specify the dataset name (e.g. MC-19-0. If unset, model will not be evaluated on data"),
   # dict(dest="--training-and-perf-path", default=None, type=str),
   # dict(dest="--eval-win-step-size", default=1, type=int),
   # dict(dest="--pred-inspect-eval", default=False, action='store_true'),
    dict(dest="--base-output-path", default=None, type=str),
   # dict(dest="--eval-filter", default=None, type=str),
    dict(dest='--device', default='cuda:0'),
    dict(dest='--stream-type', default='ecog'),
    dict(dest='--max-buflen', default=1024),
]
if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Result Parsing")
    m_options = parser.parse_args()
    m_results = deploy_to_lsl(m_options, m_options.result_file,
                              stream_type=m_options.stream_type, max_buflen=m_options.max_buflen)
