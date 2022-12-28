# Joint Decision Models for Bitrates and Handovers

## RL Models (PPO)

This is an easy TensorFlow implementation of Pensieve [1]. 
We trained Pensieve via PPO rather than A3C.

It's a stable version, which has already prepared the training set and the test set.

You can run the repo via typing:

```
python train.py
```

instead. Results will be evaluated on the test set (from HSDPA) every 300 epochs.

### ~~MPC (MRSS)~~
???

### ~~The Reference Model~~
This model is the initial PPO model that is applied to the Pensieve problem.
The location of this model is `src/rl_reference` and you can run the model in that folder as follows: `
python3 train.py`.

### Tensorboard

During the training process, we can leverage Tensorboard for monitoring current status.

```
python -m tensorboard.main --logdir [Results folder]
```

## MPC Models


## [Reference]
### Folders
* `sat_data/`: contains the satellite traces
* `video_data/`: contains the sample videos' chunk size
### Updates
* Dec. 28, 2022 (KJ Park): Summarized the codes and wrote the Readme.
