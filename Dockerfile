FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Guide on how to use poetry to install stuff in a container coming from here:
# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker

ARG YOUR_ENV=production

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.5.1

WORKDIR /workspace

EXPOSE 8000

COPY poetry.lock pyproject.toml /workspace/

RUN pip install "poetry==$POETRY_VERSION"

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# Source code
ADD app/ ./app
ADD scripts/ ./scripts
ADD src/ ./src

# DEBUG STUFF FROM HERE ON!!!
COPY data/gtzan/metal/metal.00001.wav ./

COPY lightning_logs/version_16/checkpoints/epoch=49-step=650.ckpt /workspace/lightning_logs/version_16/checkpoints/epoch=49-step=650.ckpt


# CMD [ "python", "-c", "import torch" ]
CMD [ "uvicorn", "--app-dir", "app", "--host=0.0.0.0", "main:app" ]