# Machine Learning Prediction in Real Time Using Docker and Python REST APIs with Flask

In this project, I want to show how to deploy a machine learning algorithm with a Python-Flask RESTful-API around a localhost Docker container and get real-time online predictions.

To learn this concept, I will implement online inferences (Linear Discriminant Analysis and Multi-layer Perceptron Neural Network models) with Docker and Flask-RESTful.

I found a pre-trained engine on EEG data that predicts the alphabet letter the human subject had thought.

This guide wrote for Windows Terminal and if you have another OS you should change it if you need.

In this guide I got some codes from @xaviervasques. 

## Install requirements

Before we start, we should [install Docker](https://docs.docker.com/desktop/) on our computer. If you have Windows OS, it would be better to activate and install [Windows Subsystem for Linux (WSL) Version 2](https://docs.microsoft.com/en-us/windows/wsl/install) and then ubuntu distribution on it.

After installation of Docker, we need to install [Python](https://hub.docker.com/_/python), [Scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) and [curl](https://hub.docker.com/r/curlimages/curl) on Docker.

``` bat
docker pull python
docker pull jupyter/scipy-notebook
docker pull curlimages/curl
```

The next step is to clone [ML online prediction repository](https://github.com/magnooj/ml-online-prediction) from GitHub.

``` bat
git clone https://github.com/magnooj/ml-online-prediction.git
cd ml-online-prediction
```

## Through the files

Now, we are ready to run and test our project. By running `ls` you can see these files:

- [`api.py`](https://github.com/magnooj/ml-online-prediction/blob/main/api.py) : Python-Flask RESTful-API
- [`Dockerfile`](https://github.com/magnooj/ml-online-prediction/blob/main/Dockerfile) : Docker container
- [`README.md`](https://github.com/magnooj/ml-online-prediction/blob/main/README.md) : Instructions
- [`requirements.txt`](https://github.com/magnooj/ml-online-prediction/blob/main/requirements.txt) : Required modules to run this project
- [`test.json`](https://github.com/magnooj/ml-online-prediction/blob/main/test.json) : Dataset for test the app
- [`train.csv`](https://github.com/magnooj/ml-online-prediction/blob/main/train.csv) : Dataset for train the app
- [`train.py`](https://github.com/magnooj/ml-online-prediction/blob/main/train.py) : Machine Learning app

The ``train.py`` is a python script that ingest and normalize EEG data and train two models to classify the data. The `Dockerfile` will be used to build our Docker image, `requirements.txt` *(flask, flask-restful, joblib)* is for the Python dependencies and `api.py` is the script that will be called to perform the online inference using *REST APIs*. `train.csv` are the data used to train our models, and `test.json` is a file containing new EEG data that will be used with our inference models.

### ***Flask RESTful APIs***

The first step in building *APIs* is to think about the data we want to handle, how we want to handle it and what output we want with our *APIs*. In our example, we will use the `test.json` file in which we have 1300 rows of EEG data with 160 features each (columns). We want our *APIs* to the following:

- ***API 1***: We will give a row number to the API which will extract for us the data from the selected row and print it.
- ***API 2***: We will give a row number to the API which will extract the selected row, inject the new data into the models and retrieve the classification prediction (Letter variable in the data).
- ***API 3***: We will ask the API to take all the data in the test.json file and instantly print us the classification score of the models.

At the end, we want to access those processes by making an HTTP request.

## Run APIs through a Docker container

We have all to build our [Docker Image](https://docs.docker.com/engine/reference/commandline/images/). To start, we need our `Dockerfile` with the `jupyter/scipy-notebook` image as our base image. We also need to set our environment variables and install `joblib` to allow serialization and deserialization of our trained models and flask (`requirements.txt`).

### ***Create a Docker Image***

After installation of Docker App, we should create a [Docker Image](https://docs.docker.com/engine/reference/commandline/images/). Thus, we copy the `train.csv`, `test.json`, `train.py` and `api.py` files into the image. Then, we run `train.py` which will fit and serialize the machine learning models as part of our image build process.

To build this image, run the following command:

``` bat
docker build -t my-docker-api -f Dockerfile .
```

We will get this output:

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/1.png">

We can see our images by `Docker images` command: :

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/2.png">

### ***Serve a Docker container***

Now the goal is to run our online inference meaning that each time a client issues a POST request to the `/line/<Line>`, `/prediction/<Line>`, `/score` endpoints, we will show the requested data (row), predict the class of the data we inject using our pre-trained models, and the score of our pre-trained models using all the available data. To launch the web server, we will run a Docker container and run the `api.py` script:

``` bat
docker run -it -p 5000:5000 my-docker-api python api.py
```

The `-p` flag exposes port `5000` in the container to `port 5000` on our host machine, `-it` flag allows us to see the logs from the container and we run `python api.py` in the `my-api` image.

The output is the following:

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/3.png">

You can see that we are running on http://localhost:5000/ and we can now use our web browser or the `curl` command to issue a POST request to the IP address.

If we type:

``` bat
curl http://localhost:5000/line/232
```

We will get row number 232 extracted from our data (`test.json`):

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/4.png">

Same result using the web browser:

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/5.png">

If we type the following `curl` command:

``` bat
curl http://localhost:5000/prediction/232
```

We will see the following output:

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/6.png">

The above output means that the LDA model classified the provided data (*row 232*) as letter 21 (*U*) while Multi-layer Perceptron Neural Network classified the data as letter 8 (*H*). The two models do not agree.

If we type:

``` bat
curl http://lohalhost:5000/score
```

We will see the score of our models on the entire dataset:

<img src="https://github.com/magnooj/ml-online-prediction/blob/main/images/7.png">

As we can read, we should trust more the Multi-layer Perceptron Neural Network with an accuracy score of `0.59` even if the score is not so high.

I hope you enjoyed how to containerizing your machine/deep learning applications using Docker and flask to perform online inference. if you have any comments please do not hesitate to send me an [e-mail](mailto:magnooj@gmail.com).

Regards,

Ali Ganjizadeh
