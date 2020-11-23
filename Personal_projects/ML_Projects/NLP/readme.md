# Fake News Classifier

This repository is an example of a fake news classifier utilizing NLP techniques. The repository starts with notebooks for EDA and includes a dockerised model for deployment. This repository includes:

1) notebooks - Contains longform documentation of the steps and decisions chosen in developing the workflow
2) src - Contains supporting code for the different workflow steps
3) tests - Contains tests for functions defined in src (to-do)
4) param_store - Stores important objects such as the trained model and intermediate requirements

Alongside this there is a pipfile and Dockerfile, allowing the whole process to be ran in isolation.

## Running Data_file_splitting.py

Data_file_splitting.py takes 1 user-supplied command line arguement:

- the name of the file, data_file

To run this file:

```
Python Data_file_splitting.py filename
```

This will return testing and training files.

## Running Generate_feature_columns.py

Generate_feature_columns.py takes 3 user-supplied command line arguements: 

- the name of the file, data_file

- The type of data, e.g. training or test, data_nature

To run this file:

```
Python Generate_feature_columns.py data_file data_nature
```

For training data this will apply various fit/transform methods that are saved so the transform method can be used for the test data.

## Running Feature_selection.py

Feature_selection.py takes 1 user-supplied command line arguement:

- the name of the file, data_file

To run this file:

```
python Feature_selection.py filename
```

This will produce a list of columns that will be used for modelling

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