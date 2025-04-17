# podcast-benchmark
benchmarking project for podcast decoding.

Comparing brain --> word decoding performance to [previously published results](https://www.nature.com/articles/s41593-022-01026-4)

For long updates and discussions use [this notebook ](https://docs.google.com/document/d/1IE1v_CyjZxTYaYVncxctJqZYzmYyFIgdZLXpKvEMaqc/edit?usp=sharing)

## Setup

To download data and setup your local virtual env run

```
./setup.sh
```

## Training

Currently the code is in place to train:

1. A recreation of the decoder from [https://www.nature.com/articles/s41593-022-01026-4] which decodes word embeddings directly from neural data. To train this model run:

```
make neural-conv
```

2. A decoder from our foundation model to word embeddings. To train this model run:

```
make foundation-model
```

To alter the data/behavior/hyperparameters of the training runs alter the configuration in config.py. 

## Setting up a new model
To set up a new model i.e. BrainBert there are several steps you have to do.

For quick access here are those steps:

1. Create a new folder for your model code (i.e. neural_conv_decoder/ or foundation_model/).
2. Define a decoding model and a constructor function. 
3. Define a data preprocessing function for preparing the data for your decoding model.
4. Create a config file in configs/
5. Optionally define a config setter function which lets you override config fields in code to capture config values you don't know the values of at compile time.
6. Import your models module in main.py
7. Optionally update the Makefile with a pointer to your config.
8. Try running your training code!

If you've successfully done all of the above it should just work and you can now decode from the neural data! See below for more explanation on how to do each step.

If you're hitting a weird error in the main.py script, first make sure you've properly used the function decorators and you're importing your module correctly! Details on how to do that are described below but it's a very simple mistake to make.

### 1. Create a new folder for your model code
Write all the code for a particular model in its own folder. For a model my-model run

```
mkdir my_model
```

and define all your code for this model in this new folder.

### 2. Define a decoding model and a constructor function.  

For our neural_conv_decoder model we can define our model code in neural_conv_decoder/decoder_model.py using pytorch. See PitomModel or EnsemblePitomModel. To interface with the script code you need to build a constructor function as well. A model constructor defines how to construct the model class given a provided dictionary of model parameters which you can define in your config file (see below for more details). Using the defined EnsemblePitomModel class this would like the following:

```
import registry

# Note the function decorator here! You must do the same on your own code. By default it will just set the name as
# the name of the constructor function but you can optionally change it to a name you like more.
@registry.register_model_constructor()
def ensemble_pitom_model(model_params):
    return EnsemblePitomModel(
            num_models=model_params['num_models'],
            input_channels=model_params['input_channels'],
            output_dim=model_params['embedding_dim'],
            conv_filters=model_params['conv_filters'],
            reg=model_params['reg'],
            reg_head=model_params['reg_head'],
            dropout=model_params['dropout']
        )
```

 The constructor must have the signature 

```
constructor_fn(model_params: dict) -> Model
```

where Model is whatever Model class you've defined for your use-case.

As an alternative example for our foundation model we instead use a simple MLP defined in foundation_model/foundation_decoder.py. Our constructor in this case looks like:

```
@registry.register_model_constructor()
def foundation_mlp(model_params):
    return MLP(model_params['layer_sizes'])
```

### 3. Define a data preprocessing function for preparing the data for your decoding model.

Next we need a preprocessing function to transform the neural data as we would like for this particular model. For example, for the neural_conv_decoder model we want to average over our data so that we have effectively a lower sample rate. Our preprocessing function looks like this:

```
# Note we use another function decorator here!
@registry.register_data_preprocessor()
def preprocess_neural_data(data, preprocessor_params):
    return data.reshape(data.shape[0], data.shape[1], -1, preprocessor_params['num_average_samples']).mean(-1)
```

More generally a data preprocessor function must have the following signature:

```
preprocessing_fn(data: np.array of shape [num_words, num_electrodes, timesteps],
                            preprocessor_params: dict)  -> array of shape [num_words, ...]
```

Where:
        - data: A NumPy array of shape [num_words, num_electrodes, timesteps]
        - preprocessor_params: User-defined parameters for preprocessing (defined in your config file, see below for details)
        - Returns: A NumPy array whose first dimension is still `num_words`, 
                   but the remaining shape is arbitrary (e.g., features for model input)

The output of this function will be batched and passed directly into your decoder model during training.

This function is where you would gather any latent representations from an external model that you would like to decode from. For example for the foundation model we have the following:

```
@registry.register_data_preprocessor()
def foundation_model_preprocessing_fn(data, preprocessor_params):
    # First load the model and set it in eval mode.
    ecog_config = create_video_mae_experiment_config_from_file(os.path.join(preprocessor_params['model_dir'], "experiment_config.ini"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(ecog_config)
    model.load_state_dict(torch.load(os.path.join(preprocessor_params['model_dir'], "model.pth"), weights_only=True))
    model = model.to(device)

    model.eval()

    data_config = ecog_config.ecog_data_config
    data = data.reshape(data.shape[0], data.shape[1], -1, data_config.original_fs // data_config.new_fs)
    data = data.mean(-1)
    
    for i in range(64):
        channel = "G" + str(i + 1)
        if not np.isin(channel, preprocessor_params['ch_names']):
            data = np.insert(data, i, np.zeros_like(data[:, i, :]), axis=1)

    # Reshape to [num_examples, frequency bands (currrently 1), time, num_electrodes]
    data = np.einsum('bet->bte', data).reshape(data.shape[0], data.shape[2], 8, 8)
    data = np.expand_dims(data, axis=1)

    # Construct input dataset
    batch_size = preprocessor_params['foundation_model_batch_size']
    foundation_embeddings = []
        
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
            batch_embeddings = model(batch, forward_features=True)  # Shape: [batch_size, 16]
            foundation_embeddings.append(batch_embeddings.cpu().numpy())
    
    foundation_embeddings = np.vstack(foundation_embeddings)

    return foundation_embeddings
```

where I pass all of the data through my foundation model and then pass those latents to the decoder model in the shape [num_words, latent_dim].

### 4. Create a config file in configs/

Our configuration classes are defined in config.py so see there for more details.

On a large scale our configuration files should follow this form:

```
@dataclass
class ExperimentConfig:
    # Model constructor function name. Must be registered using @registry.register_model_constructor()
    model_constructor_name: str = ''
    # Config setter function name. Must be registered using @registry.register_config_setter()
    config_setter_name: Optional[str] = None
    # Parameters for this model. Can be any user-defined dictionary.
    model_params: dict = field(default_factory=lambda: {})
    # Parameters for training.
    training_params: TrainingParams = field(default_factory=lambda: TrainingParams())
    # Parameters for data loading and preprocessing. Sub-field preprocessor_params can be set for your use-case.
    data_params: DataParams = field(default_factory=lambda: DataParams())
    # Name for trial. Will be used for separating results in storage.
    trial_name: str = ''
```

Where TrainingParams and DataParams are classes also defined in config.py. Our config files are stored in yaml in the configs/ folder. For example this is what it looks like for our neural_conv_decoder model:

```
model_constructor_name: ensemble_pitom_model # this must match the name from the registry of your function!
config_setter_name: neural_conv              # this must match the name from the registry of your function!
model_params:
  conv_filters: 128
  reg: 0.35
  reg_head: 0
  dropout: 0.2
  num_models: 10
  embedding_dim: 50
training_params:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  n_folds: 5
  min_lag: 0
  max_lag: 1
  lag_step_size: 10
data_params:
  # Cross model data_params
  data_root: data
  embedding_type: gpt-2xl
  embedding_layer: 24
  embedding_pca_dim: 50
  window_width: 0.625
  preprocessing_fn_name: preprocess_neural_data  # this must match the name from the registry of your function!
  subject_ids: [1, 2, 3]
  # neural_conv_decoder specific config
  preprocessor_params:
    num_average_samples: 32
trial_name: ensemble_model_10
```

You can create a new folder in configs/ for your model and then define config.yml's in that file as you see fit.

While most of these fields are documented in more detail in config.py the main fields that you will have to define for your model are:

1. model_params
2. data_params.preprocessor_params

The model_params will be passed into your model constructor function to build your decoder model. Feel free to create any fields you need since this will be passed your defined function.

data_params.preprocessor_params will be passed into your preprocessor function to process the data. Again what fields are here are entirely up to you.

### 5. Optionally define a config setter function which lets you override config fields in code to capture config values you don't know the values of at compile time.

Sometimes we don't know all of the configuration fields that we would like to use for our model until run-time. For example for the neural_conv_decoder model we need to know how many electrodes we are training over to build our model correctly, but this value is dependent on the subjects we train over and which electrodes we care about. Rather than worry about keeping several fields in alignment we can just set our config to grab the subjects and electrodes we care about and then use a config setter function to find the number of electrodes at run time. For our neural_conv_decoder model this looks like:

```
# Another registry, don't forget!
@registry.register_config_setter('neural_conv')
def set_config_input_channels(experiment_config: ExperimentConfig, raws: list[mne.io.Raw], _df_word: pd.DataFrame, _word_embeddings: np.array) -> ExperimentConfig:
    num_electrodes = sum([len(raw.ch_names) for raw in raws])
    experiment_config.model_params['input_channels'] = num_electrodes
    return experiment_config
```

We pass in all of our relevant data to the config setter so that it can gather whatever information it needs.

More generally a config setter function must have the following signature:

```
config_setter(
            experiment_config: ExperimentConfig,
            raws: list[mne.io.Raw],
            df_word: pd.DataFrame,
            word_embeddings: np.ndarray
        ) -> ExperimentConfig
```

Where:
    - experiment_config: ExperimentConfig dataclass
    - raws: List of MNE Raw objects (continuous iEEG/EEG data)
    - df_word: DataFrame containing word-level metadata (e.g., onset times, labels)
    - word_embeddings: NumPy array of embeddings corresponding to the words

As an alternative example for the foundation model I need to know all of the channel names when preprocessing the data as well as wanting to set the window width to whatever the model expects from its pretraining. This would give us the following:

```
@registry.register_config_setter('foundation_model')
def foundation_model_config_setter(experiment_config: ExperimentConfig, raws, _df_word, _word_embeddings) -> ExperimentConfig:
    ch_names = sum([raw.info.ch_names for raw in raws], [])
    preprocessor_params = experiment_config.data_params.preprocessor_params
    preprocessor_params['ch_names'] = ch_names

    # Set window width to whatever the sample length of the foundation model is.
    ecog_config = create_video_mae_experiment_config_from_file(os.path.join(preprocessor_params['model_dir'], "experiment_config.ini"))
    experiment_config.data_params.window_width = ecog_config.ecog_data_config.sample_length
    return experiment_config
```

This isn't necessarily required for your use case but is there if it's needed.

### 6. Import your models module in main.py
This part is simple but required or else your functions won't be discovered.

First! Make sure you've properly added the function decorators to your model constructor, data preprocessor, and config setter if it exists! If you don't do that, your code will crash!

Now we need to import the code to main.py so it will be discovered when the script runs. Thankfully, I've added a very simple way to do this. If your code exists in the folder my_model/ you will do the following in main.py.

```
# Import modules which define registry functions. REQUIRED FOR ANY NEW MODELS.
import_all_from_package('neural_conv_decoder')
import_all_from_package('foundation_model')
# Add your model import here!
import_all_from_package('my_model')
```

### 7. Optionally update the Makefile with a pointer to your config.
If you want to make it easier to run the script just add a new rule to the Makefile for your model. If your config is defined in configs/my_model/my_model.yml it could look like this:

```
my-model:
	mkdir -p logs
	$(CMD) main.py \
		--config configs/my_model/my_model.yml
```

and that's it! You're now ready to run, assuming you can run the $(CMD) in your Makefile.

### 8. Try running your training code!
If you've successfully defined a Make rule now all you need to do is run:

```
make my-model
```

and you should see results populate in your results/ folder. If you are running over slurm you can debug your script using the logs in logs/ or run locally or on an interactive node by changing $(CMD). Let me know if you have any questions!
