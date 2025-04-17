import registry

@registry.register_data_preprocessor()
def preprocess_neural_data(data, preprocessor_params):
    return data.reshape(data.shape[0], data.shape[1], -1, preprocessor_params['num_average_samples']).mean(-1)


@registry.register_config_setter('neural_conv')
def set_config_input_channels(experiment_config: dict, raws, _df_word, _word_embeddings):
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config['model_params']['input_channels'] = num_electrodes
    return experiment_config
