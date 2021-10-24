# CMU-11685-HW1P2

Fall 2021 Introduction to Deep Learning - Homework 1 Part 2

Author: ryanquinnnelson

## Summary

`octopus` is a python module that standardizes the execution of deep learning pipelines using `pytorch`, `wandb`,
and `kaggle`.

```
               _---_
             /       \
            |         |
    _--_    |         |    _--_
   /__  \   \  0   0  /   /  __\
      \  \   \       /   /  /
       \  -__-       -__-  /
   |\   \    __     __    /   /|
   | \___----         ----___/ |
   \                           /
    --___--/    / \    \--___--
          /    /   \    \
    --___-    /     \    -___--
    \_    __-         -__    _/
      ----               ----
 
        O  C  T  O  P  U  S
```

## Run Requirements

- wandb account: wandb is used for model tracking and early stopping statistics. There is currently no way to turn off
  this behavior.
- kaggle account: `octopus` is configured to download data from kaggle via the api. This behavior can be turned off in
  the configs.
- kaggle.json file: If `octopus` is going to download data from kaggle, it expects a kaggle token file `kaggle.json` to
  be available. The file must contain your kaggle username and api key in the standard JSON format.
- configuration file: The file is expected to be in a format that python's `configparser` understands. See example
  below.
- `customized` python module with one file `customized.py` that implements four classes. `octopus` comes with an example
  customized file.
    - `TrainValDataset`
    - `TestDataset`
    - `OutputFormatter`
    - `Evaluation`

#### TrainValDataset

Defines Training and Evaluation datasets for your data.

- Subclass of `torch.utils.data.Dataset`
- Implements standard Dataset methods: `__init__()`, `__len__()`, `__getitem__()`

#### TestDataset

Defines Testing dataset for your data.

- Subclass of `torch.utils.data.Dataset`
- Implements standard Dataset methods: `__init__()`, `__len__()`, `__getitem__()`

#### OutputFormatter

Defines how output from the test set should be formatted to meet desired requirements (i.e. Kaggle submission).

- Output from `format()` will be saved to file.
- Implements: `format(self, out) -> DataFrame`

#### Evaluation

Defines the evaluation process for each epoch.

- Implements: `__init__(self, val_loader, criterion_func, devicehandler)`
- Implements: `evaluate_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`
- Note that val_metric is a generic name. Match the actual metric returned in that position to the config file.

## Steps to Run

Module requires a single argument - the path to the configuration file. Module expects `pytorch`, `numpy`, and `pandas`
to be available in the environment.

```bash
$ run_octopus.py /path/to/config.txt
```

## Configuration file requirements

The following is an example of a configuration file. All configs are required. Values are fully customizable.

```text
[DEFAULT]
run_name = Run-07

[debug]
debug_file = /Users/ryanqnelson/Desktop/test/debug.log

[kaggle]
download_from_kaggle = True
kaggle_dir = /Users/ryanqnelson/Desktop/test/.kaggle
content_dir = /Users/ryanqnelson/Desktop/test/content/
token_file = /Users/ryanqnelson/Desktop/kaggle.json
competition = hw1p2-toy-problem
delete_zipfiles_after_unzipping = True

[wandb]
wandb_dir = /Users/ryanqnelson/Desktop/test
entity = ryanquinnnelson
project = CMU-11685-HW1P2-octopus-4
notes = Simple MLP
tags = MLP,octopus

[stats]
# metric for comparison against previous runs to check whether early stopping should occur
comparison_metric=val_loss

# if a lower value is better for the metric, set comparison_best_is_max to False
comparison_best_is_max=False

# how many epochs to allow the current model to perform worse than the best model for this metric before stopping
comparison_patience=20

# name of the metric that evaluate_model() returns in the second position of the tuple, for stats purposes
val_metric_name=val_acc

[data]
data_dir = /Users/ryanqnelson/Desktop/test/content/competitions/hw1p2-toy-problem
output_dir = /Users/ryanqnelson/Desktop/test/output
train_data_file=toy_train_data.npy
train_label_file=toy_train_label.npy
val_data_file=toy_val_data.npy
val_label_file=toy_val_label.npy
test_data_file=toy_test_data.npy

# input is (2 * context + 1) * 40 for this dataset
input_size=200
output_size=71

[model]
model_type=MLP

[checkpoint]
checkpoint_dir = /Users/ryanqnelson/Desktop/test/checkpoints
delete_existing_checkpoints = True
load_from_checkpoint=False

# if loading from a checkpoint, supply a fully qualified filename for the checkpoint file
checkpoint_file =  None

[hyperparameters]
dataloader_num_workers=8
dataloader_pin_memory=True
dataloader_batch_size=128

# keyword arguments to be supplied to Dataset subclasses when loading data
dataset_kwargs=context=2
num_epochs = 3
hidden_layer_sizes = 128
dropout_rate=0.0
batch_norm=False
activation_func=ReLU
criterion_type=CrossEntropyLoss
optimizer_type=Adam

# keyword arguments to be supplied when initializing the optimizer
optimizer_kwargs=lr=0.001
scheduler_type=ReduceLROnPlateau

# keyword arguments to be supplied when initializing the scheduler
scheduler_kwargs=factor=0.1,patience=5,mode=max,verbose=True

# name of the metric stored in stats that the scheduler uses (if necessary) to determine its step
scheduler_plateau_metric=val_acc
```

## Additional Features

Repository also includes a bash script `mount_drive.sh` which will mount a drive on an AWS EC2 instance and prepare it
for use.

### How to run

```bash
$ mount_drive.sh
```

## Improvement Ideas

- ~~Multiple configuration files~~ (Implemented in `octopus` in CMU-11685-HW2P2)
    - Current: `octopus` executes the pipeline for a single configuration file.
    - Improvement: If multiple configuration files are placed into a configuration folder, `octopus` will execute the
      pipeline for each configuration file.
- ~~wandb checkpoint metrics~~ (Implemented in `octopus` in CMU-11685-HW2P2)
    - Current: When loading from checkpoint (and therefore starting from an epoch other than 1), the metrics sent to
      wandb will appear shifted relative to runs that sent metrics from epoch 1. For a previous run, the metric for
      epoch 1 is displayed as position 0 on wandb. For the checkpoint run, the metric for epoch X is displayed as
      position 0 on wandb.
    - Improvement: wandb shifts over and displays the metrics from the checkpointed run in the correct position in its
      graphs. A few ways to do this: (1) If loading from a checkpoint, `octopus` sends dummy metrics to wandb for each
      of the epochs before the one that the pipeline starts on; (2) `octopus` sends the actual metrics from all epochs
      stored in the checkpoint.
- ~~weight initializations~~ (Implemented for Resnet in `octopus` in CMU-11685-HW2P2)
    - Current: model uses default initializations
    - Improvement: implement Kaiming Initialization or Xavier Initialization
- Ability to turn off wandb
    - Current: there is no way to turn off the service because many parts of `octopus` use wandb.
    - Improvement: a single change in the config file stops use of wandb throughout the pipeline.
- mount script arguments
    - Current: mount script assumes the name of the drive to mount.
    - Improvement: we pass in the name of the drive to mount as an argument to the script.

## How octopus is used for this project

We were given a dataset of audio recordings (utterances) and asked to classify each speech frame into one of 71 phoneme
states. We were given 14542 training utterances (18482968 frames), 2683 validation utterances (1935669 frames), and 2600
test utterances (1910012 frames). Audio utterances were already preprocessed into raw melspectrograms frames. Utterances
had a variable number of frames, but every frame had 40 dimensions.

We were asked to implement a multilayer perceptron (MLP) model to classify the frames. One of the important
hyperparameters to tune was the amount of context provided around each frame. On its own, a single frame doesn't contain
much information. High accuracy was only achievable by including some number of surrounding frames for context.

I customized the `customized` module for this dataset and formatted output for the kaggle competition. The highest
accuracy I was able to achieve on the private kaggle leaderboard was 79.704% (rank 17/316).

