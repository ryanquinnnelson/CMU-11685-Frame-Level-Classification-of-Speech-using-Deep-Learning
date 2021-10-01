# CMU-11685-HW1P2

Fall 2021 Introduction to Deep Learning - Homework 1 Part 2

## Summary

`octopus` is a python module that standardizes the execution of deep learning pipelines using `pytorch`, `wandb`, and `kaggle`.

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

### Requirements

- configuration file: The file is expected to be in a format that python's `configparser` understands.
- kaggle.json file: The module also expects a kaggle token file `kaggle.json` to be available.
- `customized` python module with one file `customized.py` that implements four classes:
    - `TrainValDataset`
    - `TestDataset`
    - `OutputFormatter`
    - `Evaluation`

#### TrainValDataset

- Defines Training and Evaluation datasets.
- Subclass of `torch.utils.data.Dataset`
- Implements standard Dataset methods: `__init__()`, `__len__()`, `__getitem__()`

#### TestDataset

- Defines Testing dataset.
- Subclass of `torch.utils.data.Dataset`
- Implements standard Dataset methods: `__init__()`, `__len__()`, `__getitem__()`

#### OutputFormatter

- Defines how output from test should be formatted to meet Kaggle requirements. Output from `format()` will be saved to
  file.
- Implements: `format(self, out) -> DataFrame`

#### Evaluation

- Defines the evaluation process for each epoch.
- Implements: `__init__(self, val_loader, criterion_func, devicehandler)`
- Implements: `evaluate_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`
- Note that val_metric is a generic name. Match the actual metric returned in that position to the config file.

### How to run

Module requires a single argument - the path to the configuration file.

```bash
$ run_octopus.py /path/to/config.txt
```

### Configuration File requirements

The following is an example of a configuration file. All configs are required. Values are fully customizable.

```text
[DEFAULT]

[debug]
debug_file = /Users/ryanqnelson/Desktop/test/debug.log

[kaggle]
kaggle_dir = /Users/ryanqnelson/Desktop/test/.kaggle
token_file = /Users/ryanqnelson/Desktop/test/kaggle.json
competition = hw1p2-toy-problem
delete_zipfiles_after_unzipping = True

[wandb]
entity = ryanquinnnelson
name = Run-02
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
data_dir = /Users/ryanqnelson/Desktop/test/content
output_dir = /Users/ryanqnelson/Desktop/test/output
train_data_file=toy_train_data.npy
train_label_file=toy_train_label.npy
val_data_file=toy_val_data.npy
val_label_file=toy_val_label.npy
test_data_file=toy_test_data.npy
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
num_epochs = 30
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
- Multiple configuration files
  - Current: `octopus` executes the pipeline for a single configuration file.
  - Improvement: If multiple configuration files are placed into a configuration folder, `octopus` will execute the pipeline for each configuration file.
- wandb checkpoint metrics
  - Current: When loading from checkpoint (and therefore starting from an epoch other than 1), the metrics sent to wandb will appear shifted relative to runs that sent metrics from epoch 1. For a previous run, the metric for epoch 1 is displayed as position 0 on wandb. For the checkpoint run, the metric for epoch X is displayed as position 0 on wandb.
  - Improvement: wandb shifts over and displays the metrics from the checkpointed run in the correct position in its graphs. A few ways to do this: (1) If loading from a checkpoint, `octopus` sends dummy metrics to wandb for each of the epochs before the one that the pipeline starts on; (2) `octopus` sends  the actual metrics from all epochs stored in the checkpoint.
- weight initializations
  - Current: model uses default initializations
  - Improvement: implement Kaiming Initialization or Xavier Initialization




