Machine Learning Project Template

# Introduction
This repository is mlflow and luigi based machine learning project template.

It will make easy to start machine learning project regardless of dataset and framework with data preprocessing.

You can rerun at main task or intermediate preprocessing, such as image cropping.

# Usage
Run following commands for install and training mnist classifier with fully connected network.
```sh
poetry install
PYTHONPATH='.' poetry run luigi --module projects.run_once RunOnceProject --runner image_recognition_trainer --model fcnn --dataset mnistraw --param-path params.yaml --local-scheduler
```

After the running, result files are created under the mlflow directory.

You can see the results by `mlflow ui` and the running tasks by luigi server.

Please see [MLflow Document](https://www.mlflow.org/docs/latest/index.html) and [Luigi Document](https://luigi.readthedocs.io/en/stable/#) for detail usage.

## running configuration
Following parameters should be set for running.
- `module`: Only 'projects.run_once' can be set now.
- `Task Name`: Only 'RunOnceProject' can be set after module name.
- `runner`: Running target. File name under the 'runner' directory can be set.
- `model`: Running model name. Any model allowed by runner can be set.
- `dataset`: Running dataset name. Any dataset allowed by runner can be set.
- `param-path`: Model and dataset and preprocesssing parameters.

## Learning parameters configuration
Parameters configuration can set `param-path` to yaml format file.

It includes model and dataset and preprocesssing section.

### Model configuration

Model configuration is diffrent allowed parameters for model.

Model object extends [KerasClassifierBase](/model/base.py#L79) can be set number of epoch, learning rate, etc.

```yaml
model:
  epochs: 100
  lr: 0.001
```

It means that model run 100 epochs and 0.001 learning rate.

### Dataset configuration

Dataset configuration is diffrent allowed parameters for dataset.

Dataset object extends [ImageClassifierDatasetBase](/dataset/base.py#L58) can be set batch size, etc.

```yaml
dataset:
  batch_size: 128
```

It means that dataset is used with 128 batch.

### Preprocess configuration

Preprocess configuration consits of preprocess projects, these parameters and rerunning target.

`projects` is preprocess projects setting as dictionary. project name and function name pairs are set, projects are ran from top to bottom and use previous project results.

`parameters` is all project configuration. It is used to every preprocess project.

`update_task` is rerun target project name. When `update_task` is set, the running start target project. In contrast, when `update_task` is empty, the running only act runner project.

In any case, when dependent projects have not been acted with configuration parameters, these projects are running.

```yaml
preprocess:
  projects:
    Download:
      dataset.mnist_from_raw.download_data
    Decompose:
      dataset.mnist_from_raw.decompose_data

  parameters:
    name: sample

  update_task: 'Decompose'
```

It means that download and decompose projects are running with name parameter, but if download have not been acted.

## Workflow for your own task
This framework can apply to specific model, specific dataset and specific task.

For example, when you use to image classification task with your custom model,you implement model class extends [KerasClassifierBase](/model/base.py#L79) class.

This class is implemented to learn image classification with `model` property, 
which constructed by Keras functional API.

# Test and linting
Run following command for testing and linting.
```sh
poetry run tox
```

# ml-app-template

An ML project template with sensible defaults:
- Dockerised dev setup
- Unit test setup
- Automated tests for model metrics
- CI pipeline as code

For infrastructure-related stuff (e.g. provisioning of CI server, deployments, etc.), please refer to https://github.com/ThoughtWorksInc/ml-cd-starter-kit.

## Getting started

1. Fork repository: https://github.com/ThoughtWorksInc/ml-app-template
2. Clone repository: `git clone https://github.com/YOUR_USERNAME/ml-app-template`
3. To develop on local environment with installed Python packages, run: `pipenv install` then activate environment with `pipenv shell` 
    3.b. to run anything without activating the virtual environment, for example, nosetests, try `pipenv run nosetests`
4. Install Docker ([Mac](https://docs.docker.com/docker-for-mac/install/), [Linux](https://docs.docker.com/install/linux/docker-ce/ubuntu/))
5. Start Docker on your desktop
6. Build image and start container:

```shell
# build docker image [Mac/Linux users]
docker build . -t ml-app-template

# build docker image [Windows users]
MSYS_NO_PATHCONV=1 docker build . -t ml-app-template

# start docker container [Mac/Linux users]
docker run -it  -v $(pwd):/home/ml-app-template \
                -p 8080:8080 \
                -p 8888:8888 \
                ml-app-template bash

# start docker container [Windows users]
winpty docker run -it -v C:\\Users\\path\\to\\your\\ml-app-template:/home/ml-app-template -p 8080:8080 -p 8888:8888 ml-app-template bash
# Note: to find the path, you can run `pwd` in git bash, and manually replace forward slashes (/) with double backslashes (\\)
```

You're ready to roll! Here are some common commands that you can run in your dev workflow. Run these in the container.

```shell
# add some color to your terminal
source bin/color_my_terminal.sh

# activate virtual environment for python
pipenv shell

# run unit tests
nosetests

# run unit tests in watch mode and color output
nosetests --with-watch --rednose --nologcapture

# train model
SHOULD_USE_MLFLOW=false python src/train.py

# start flask app in development mode
python src/app.py

# make requests to your app
# 1. In your browser, visit http://localhost:8080
# 2. Open another terminal in the running container (detailed instructions below) and run:
bin/predict.sh http://localhost:8080

# You can also use this script to test your deployed application later:
bin/predict.sh http://my-app.com
```

Here are some other commands that you may find useful
```shell
# see list of running containers
docker ps

# start a bash shell in a running container
docker exec -it <container-id> bash

# starting jupyter notebook server on http://localhost:8888
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

## What's in this repo?

We've created a project template to help you with the boilerplate code that we usually have to write in any typical project.

To reduce incidental complexity, we used a simple dataset (boston housing prices) to train a simple linear regression model. Replace the (i) data, (ii) data preprocessing code and (iii) model specification for your use case.

This is the project structure:

```sh
.
├── Dockerfile
├── README.md
├── requirements-dev.txt              # specify dev dependencies (e.g. jupyter) here
├── requirements.txt                  # specify app dependencies here
├── ci.gocd.yaml                      # specify your CI pipeline here
└── src                               # place your code here
    ├── app.py
    ├── app_with_logging.py           
    ├── tests                         # place your tests here
    │   ├── test.py
    │   └── test_model_metrics.py
    └── settings.py                   # define environment variables here
    └── train.py
├── bin                               # store shell scripts here
│   ├── color_my_terminal.sh
│   ├── configure_venv_locally.sh
│   ├── predict.sh
│   ├── start_server.sh
│   ├── test.sh
│   ├── test_model_metrics.sh
│   └── train_model.sh
├── docs
│   ├── FAQs.md
│   └── mlflow.md
├── models                            # serialize stuff here
│   ├── _keep
│   ├── column_order.joblib
│   └── model.joblib

```

For logging, `app_with_logging.py` contains the code for logging (i) inputs to the model, (ii) model outputs and (iii) [LIME](https://github.com/marcotcr/lime) metrics. You can refer to this file to send logs to elasticsearch using fluentd. To keep the main app simple to accessible to people who may not be familiar with these technologies, we've kept it in a separate file `app_with_logging.py` for reference.

# Todo
- [] hyper parameter search template

# License
MIT License

