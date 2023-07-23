# Music Genre classifier assignment

## Requirements

* Docker (and compose)
* Poetry
* Python 3.10
* a GPU with at least 12GB VRAM (tested on a T4)

## Setup

### Download and extract datasets and model checkpoints

```shell
wget https://storage.googleapis.com/misc-c4l3b/epidemicsound-tech-test/gtzan_5x_augmentations.tar.gz
wget https://storage.googleapis.com/misc-c4l3b/epidemicsound-tech-test/gtzan_no_augmentations.tar.gz
wget https://storage.googleapis.com/misc-c4l3b/epidemicsound-tech-test/model_checkpoints.tar.gz
```

```shell
mkdir -p data
tar -xzvf gtzan_5x_augmentations.tar.gz --directory=data
tar -xzvf gtzan_no_augmentations.tar.gz --directory=data
tar -xzvf model_checkpoints.tar.gz
```

### Setup virtual environment

Install poetry (if not already available)

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Install python packages

```shell
poetry install
```

## Evaluate trained models

There is a script that performs the evaluation steps described in the "Results" section of the report, it can be executed using the following command

```shell
poetry run python \
    scripts/compute_metrics.py \
    DATASET_NAME \
    PATH/TO/MODEL_CHECKPOINTS_FOLDER \
    --cm-filename CM_IMAGE_FILENAME
```

Valid (existing) values for `DATASET_NAME` are `gtzan_augmented_256_x0_no_aug` (no augmentations) and `gtzan_augmented_256_x5` (with augmentations).

`PATH/TO/MODEL_CHECKPOINTS_FOLDER` should point to the folder containing the 5 checkpoints, they should match pattern `*fold_{f}*.ckpt`, with f in [0, 4].

Example (should work out of the bag if no changes have been made to the folder structure):

```shell
poetry run python \
    scripts/compute_metrics.py \
    gtzan_augmented_256_x0_no_aug \
    model_checkpoints/20230723_062258_effnet_b1_no_pretrain_noaug_1024_focal \
    --cm-filename cm_noaug_nopretrain_1024.png
```

For obvious reasons, models trained on the augmented dataset should be tested on the augmented dataset and the other way around (the names of the models' folders has tags indicating how they were trained).

## Train the model

Before starting the training, it is necessary to launch a local instance of mongodb/omniboard.

```shell
# From the repository root
docker compose up
```

Alternatively, this can [probably] be disabled by commenting the following line in `scripts/train.py`, about line 26.

```python
ex.observers.append(
    MongoObserver(url="mongodb://sample:password@localhost", db_name="db")
)
```

A bash script is provided to wrap the actual call to the python script, so that it runs the script on all folds and applies some tags as well.

```shell
./scripts/train_command.sh
```

The four bash scripts used to train the models included above are in the same folder (`scripts`); their name should be pretty self-explanatory.

## Build docker container for inference

```shell
make build
```

1/0

## Test inference

Either first deploy the dockerized version of the classifier and
then submit a request using `curl`

```shell
make serve

curl \
    -X POST 'http://localhost:8000/uploadfile' \
    --form 'file=@"path/to/wav/file.wav"'
```

or directly with the included inference script (assumes poetry 
was used for environment setup)

```shell
poetry run python scripts/inference.py path/to/wav/file.wav
```