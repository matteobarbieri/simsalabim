# Music Genre classifier

Trained for now on the GTzan dataset, 10 music genres.

# Train the model

```shell
poetry run python scripts/train.py
```

# Build docker container for inference

```shell
make build
```

# Test inference

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