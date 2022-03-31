from ecog_speech.experiments import standard, transfer_learning
all_model_hyperparam_names = standard.all_model_hyperparam_names + transfer_learning.all_model_hyperparam_names

make_model = standard.make_model
