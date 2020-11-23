# Ames Dataset Regression

This repository is an example of a ML pipeline for a simple regression task, starting with exploratory data analysis in notebooks, and finishing with a dockerised model that can be deployed to servers. This repository is composed of the following sub-directories:

1) notebooks - Contains longform documentation of the steps and decisions chosen in developing the workflow
2) src - Contains supporting code for the different workflow steps
3) tests - Contains tests for functions defined in src (to-do)
4) param_store - Stores important objects such as the trained model and intermediate requirements

Alongside this there is a pipfile and Dockerfile, allowing the whole process to be ran in isolation.

## Using pipfile

Running

```console
pipenv install
```

in the root directory will install the required packages. Then running

```console
pipenv run jupyter-notebook
```

will launch a notebook within the pipenv environment, given access to the required packages.

## Using Dockerfile

Running

```console
docker build -t image_name Dockerfile .
```

will execute the dockerfile, creating an isolated container. To access this container run

```console
 docker run -it --rm -v "{DIRECTORY}:/app"  ames /bin/bash
```

to enter into a bash environment within the container. From here the top level scripts can be executed via python CLI calls.