# CMU-11685-HW1P2
Fall 2021 Introduction to Deep Learning - Homework 1 Part 2

## Summary
`octopus` is a python module that standardizes the execution of deep learning pipelines using pytorch.
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
- kaggle.json file: The module also requires a kaggle token file `kaggle.json` to be available.


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
debug_file = /home/ubuntu/debug.log

[kaggle]
kaggle_dir = /home/ubuntu/.kaggle
token_file = /home/ubuntu/kaggle.json
competition = idl-fall2021-hw1p2
delete_zipfiles = True

[wandb]
entity = ryanquinnnelson
name = Run-03
project = CMU-11685-HW1P2-octopus-3
notes = Simple MLP
tags = MLP,octopus

[earlystop]
comparison_metric=val_acc
comparison_best_is_max=True
comparison_patience=5

[data]
data_dir = /home/ubuntu/content
output_dir = /home/ubuntu/output
train_data_file=train.npy
train_label_file=train_labels.npy
val_data_file=dev.npy
val_label_file=dev_labels.npy
test_data_file=test.npy

# input is (2 * context + 1) * 40
input_size=1640
output_size=71

[model]
model_type=MLP

[checkpoint]
checkpoint_dir = /home/ubuntu/checkpoints
delete_existing_checkpoints = True
load_from_checkpoint=False
checkpoint_file = None

[hyperparameters]
dataloader_num_workers=8
dataloader_pin_memory=True
dataloader_batch_size=128
dataset_kwargs=context=20
num_epochs = 20
hidden_layer_sizes = 2048,2048,2048,1024,512
dropout_rate=0.3
batch_norm=True
activation_func=ReLU
criterion_type=CrossEntropyLoss
optimizer_type=Adam
optimizer_kwargs=lr=0.001
scheduler_type=ReduceLROnPlateau
scheduler_kwargs=factor=0.1,patience=5,mode=max,verbose=True
scheduler_plateau_metric=val_acc
```

## Additional Features
Repository also includes a bash script `mount_drive.sh` which will mount a drive on an AWS EC2 instance and prepare it for use.

### How to run
```bash
$ mount_drive.sh
```
