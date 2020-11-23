FROM python:3.7.8

WORKDIR app

COPY Pipfile .
COPY Pipfile.lock .

RUN python -m pip install --upgrade pip && \
    python -m pip install pipenv==2018.11.26 && \
    pipenv install --system --keep-outdated --ignore-pipfile --dev
