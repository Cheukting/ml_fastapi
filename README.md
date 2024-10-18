# Deploy your Machine Learning model with Fast API

In this workshop, we will go deeper into how to prototype a machine-learning project with [Fast API](https://fastapi.tiangolo.com/). Fast API allows the creation API server with very little effort, it is easy to deploy a pre-trained model, but for models that require re-training, the challenge of when and how to retrain a model and update for a service in use becomes complicated. We will cover the aspect of delivering a pre-trained model and the design of re-training the model. This workshop will also provide suggestions for deploying the machine learning project so it can migrate from a prototype to a functional service in production.

## Prerequisite

This workshop assume that you have experience code in Python and have knowledge using some of the data science and machine learning library such as [pandas](https://pandas.pydata.org/docs/index.html), [Scikit-learn](https://scikit-learn.org/stable/index.html) and [Keras](https://keras.io/). Details explaining usage of those libraries will be skip in this workshop

## Preflight check

Please make sure you are using Python 3.12, this is the Python version that we will be using. You may try using other version of Python but we will not guarantee all exercises will work the same.

If you want to complete part 3 of the workshop, you will need to be able to deploy [docker containers](https://www.docker.com/) locally.

## Installation

Requirements are in the file `requirements.txt` [here](requirements.txt), we recommend using [uv](https://github.com/astral-sh/uv) to create a new environment and install dependencies.

We also recommend install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for part 3 of the workshop.  

---

## Part 1 - Introduction to Fast APi and prediction on demand

In this part, we will keep things simple and just go over the simplest usage of Fast API.

### What is Fast API?

FastAPI is a web framework for building APIs with Python based on standard Python type hints. It allows you to build an API application with short amount of time. Fast API also comes with a command line tool that let you run your application and automatically generate a [Swagger UI](https://github.com/swagger-api/swagger-ui) documentation for your application APIs. We will explore these tools in the workshop.

### Why Fast API for machine learning?

A lot of data science and machine learning team do not have the resources to focus on building an application for their project. Fast API allows the team to build a prototype application quickly, but also with potential to deploy as a application in production if needed. We will go through some of the tools that can be used for building a prototype application in this workshop.

### Exercise 1: Fast API basics

Before we use Fast API on a machine learning project, let's get through the basics and make a "Hello World" app.

First create a file named `main.py` as below:

```python
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/users/{username}")
def read_user(username: str, q: Union[str, None] = None):
    return {"user": username, "q": q}
```

As you can see, we use `FastAPI` to create an application. All the endpoints are created as functions with our decorators in our application. In the app above, we have only used `GET` endpoints.

Now, let's give this a try and run the application using the command line tool provided by Fast API. In the terminal:

```
fastapi dev main.py
```

In the prompt, you will see:

```
╭────────── FastAPI CLI - Development mode ───────────╮                        
│                                                     │                        
│  Serving at: http://127.0.0.1:8000                  │                        
│                                                     │                        
│  API docs: http://127.0.0.1:8000/docs               │                        
│                                                     │                        
│  Running in development mode, for production use:   │                        
│                                                     │                        
│  fastapi run                                        │                        
│                                                     │                        
╰─────────────────────────────────────────────────────╯    
```

As you see, we are now running our applicaiton in development mode. There is a server running locally at `http://127.0.0.1:8000` and the automatically generate documentation is at `http://127.0.0.1:8000/docs`. To test it out, let's open a browser and goes to the url http://127.0.0.1:8000.

Now, you will see the "Hello World" message that is provided at the root. So everything seems fine so far. Let's also test out the user endpoint with this url http://127.0.0.1:8000/users/johndoe

You should see this response:

```
{"user":"johndoe","q":null}
```

Since we do not put in a query `q`, it is shown as `null`, now let's try to add some query:

http://127.0.0.1:8000/users/johndoe?q=somequery

You should see this response:

```
{"user":"johndoe","q":"somequery"}
```

From here, you can see how the path parameter `username` and the optional query parameter `q` work.

Before we move on, let's also check the documentation: http://127.0.0.1:8000/docs

Try playing around, espeically the user endpoint. By clicking `Try it out`, you can put in the parameters and see what the response of that endpoint. This interactive documentation is provided by [Swagger UI](https://github.com/swagger-api/swagger-ui). For the latest version of Fast API, it also provide documentation by [ReDoc](https://github.com/Rebilly/ReDoc), the link to that documentation will be http://127.0.0.1:8000/redoc, feel free to try it out as well.

### Exercise 2: prediction on demand

Now, let's consider a machine learning project using Scikit-learn. In this [Jupyter notebook](https://github.com/Cheukting/FastAPI-ml-demo/blob/main/penguins_predict.ipynb) we use a `KNeighborsClassifier` to determine the penguin species by the measurement of its bill and flipper length. The data we used can be found [here](data/), you can download the csv file and put it in a folder names `data`.

Spend a couple of minutes to look through the notebook and understand what the code does. We will skip the detail explanation of using Scikit-learn in this workshop. Also feel free to download the run the notebook for further inspection.

To create an application based on this project, let first do something basic and simple. Here is the plan: Let's do the data preparation and training of the model while the application got deploy (e.g. when we run the command `fastapi dev main.py` or `fastapi run`) while the input parameter, we can use query parameters, being the bill and flipper length. When a `GET` request is send to our prediction endpoint, we will used the trained model to give a prediction and the result will be pass in the dictionary in the response.

Let's start with putting the machine learning code for training the model in a new `main.py`:

```python
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/penguins.csv')
data = data.dropna()

le = preprocessing.LabelEncoder()
X = data[["bill_length_mm", "flipper_length_mm"]]
le.fit(data["species"])
y = le.transform(data["species"])
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)
clf.fit(X, y)

```

Here we clean the data with pandas [dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna), then create an encoder to [label encode](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) the target, which is the species of the penguins. After label encoding the target, we build a model with a [standard scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) and then using the [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

We use all the data we have for the fitting, normally we should do a [train-test-split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) and use the training data for training and testing to evaluate the performance of the model. For simplicity sake we assume experiments has been done and we are safe to use all data for training the model.

Next, we will create the predict endpoint:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/predict/")
def predict(bill: float, flipper: float):
    param = {
              "bill_length": bill,
              "flipper_length": flipper
            }
    result = clf.predict([[bill, flipper]])
    return {
        "parameters": param,
        "result": le.inverse_transform(result)[0],
    }
```

Now, we can run our application (remember what we did in the last exercise?) and test out the endpoint. You can use a direct url with the query parameter `bill` and `flipper` (for example http://127.0.0.1:8000/predict/?bill=20&flipper=200) or you can use the interactive documentation to test it.

### Exercise 3: query parameters validation

You may have a question, what if the user make a bad query, for example, putting non numbers for `bill` and `flipper`? Why not give it a try yourself?

Here comes the Fast API query parameter validation, which is power by [Pydantic](https://docs.pydantic.dev/) ([data validation is handled by Pydantic in Fast API](https://fastapi.tiangolo.com/features/#pydantic-features)). Since in our code:

```python
def predict(bill: float, flipper: float):
  ...
```

We have annotated that `bill` and `flipper` can only be float, if someone make a query otherwise it will automatically handle the error for you.

However, sometime just the type validation is not enough. Let's try to put in a negative `bill` or `flipper` (or both) value.

In our case, it is not reasonable if input of `bill` and `flipper` is less than or equal to zero. So, to do that, let's modify our code:

```python
...

from typing import Annotated

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/predict/")
def predict(
    bill: Annotated[float, Query(gt=0)],
    flipper: Annotated[float, Query(gt=0)]
):
  ...
```

Try again with negative `bill` or `flipper` value, now you will see a nice error message that comes back instead of trying to run the prediction using the model on an unreasonable value.

Before we move on, why not have a look at what else you can do for the parameter validation [here](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/) and [here](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/) and try adding a maximum value for both `bill` and `flipper`.


### Extra challenge

If you want an extra challenge, how about letting the user to input `bill` and `flipper` with units (something like "120mm") as input instead? What modification would you do to make it possible?

---

## Part 2 - Re-train and update models

So far we have a very simple machine learning model to train while the application is deployed. Things can get tricky if it is a more complex model and take more time to train. We will look at that in this part.

### When shall we train the model?

Training a more complex model, for example a deep learning model could take some time. While it is ok to train the model when we deploy the application, we may want more control over then it is trained and may not want the training of the model to block the other application. This is when the background task can be helpful.

By making use of background task, we have to design our application in asynchronous code and concurrency. This allows the best use to the operation time and didn't create unsecured blocker in our application.

In this following exercise, we will explore different strategies that can be used when handling training a machine learning model.

### Exercise 1: background task

Let's consider a deep learning model. Below is a MNIST classification model built in Keras, it is based on [the example here](https://keras.io/examples/vision/mnist_convnet/).

```python
import numpy as np
import keras
from keras import layers

# MNIST model meta parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 15

# Load the data and split it between train and test sets
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

```
With this code as `main.py` we can run `fastapi dev main.py` again. As you see, while the training is running, there is no response at the root http://127.0.0.1:8000 if you try opnining it on a browser.

Now, we will use [background task](https://fastapi.tiangolo.com/reference/background/) to perform training, before we do that, let's wrap the training in a function and put some global variables there to keep track of things:

```python
import numpy as np
import keras
from keras import layers

ml_models = []
num_train = 0

def training():
    global num_train
    num_train += 1

    # MNIST model meta parameters

    ...

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    ml_models.append(model)

    num_train -= 1

...
```

We would like to keep all the trained models in a list and keep counting the number of models in training. Once the training is done, we will add the trained model and minus the count of model in training by 1.

Next, let us look at the API endpoints:

```python

from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/info/")
async def read_models():
    return {
        "Num of model in training" : num_train,
        "Num of trained models": len(ml_models)
    }

@app.get("/train/")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(training)
    return {"message": "Training model in the background"}

```

We added a new endpoint for training the mode, This new `train` endpoint used the `BackgroundTasks` provided by `fastapi` and we added out training function to this task once this endpoint received a request.

We also added an `info` endpoint to look at what is going on, such as how many models are in training and how many are trained.

Now, let's run the application again and test out the endpoints. Observe what happened and answer the follow questions:

- Are the endpoints available soon after the application is deployed? What is the difference between this and the previous deployment?
- After requesting the `train` endpoint, what result will you get when you request the `info` endpoint?
- While the training is happening, what happen if you request the `train` endpoint again? What does the `info` endpoint tell you?

This gives you an idea of how concurrency works and also foreshadowing an issue called race condition which can happen when we are using async code. We will look at how we can deal with it later.

### Exercise 2: lifespan events

Now we know how to use background tasks but this application is not very useful since we do not have a prediction endpoint yet. This application is an image classification so for the input of the prediction, we will have to provide an image. We can do this by using [UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/#file-parameters-with-uploadfile) from `fastapi` and a POST endpoint. We will also use [pillow library](https://pillow.readthedocs.io/en/stable/) for image processing:

```python
from fastapi import UploadFile, BackgroundTasks, FastAPI

...

from PIL import Image
from io import BytesIO

@app.post("/predict/")
async def predict(file: UploadFile):
    image = await file.read()
    # process image for prediction
    image = Image.open(BytesIO(image)).convert('L')
    image = np.array(image).astype("float32") / 255
    image = np.expand_dims(image, (0, -1))
    # predict the result
    result = ml_models[-1].predict(image).argmax(axis=-1)[0]
    return {"filename": file.filename,
            "result": str(result)}
```

Note that we are using the last trained model for the prediction here, assuming it is the most updated model.

You can now test out this new endpoint with the interactive documentation. You can use any of the image I have prepared [here](data/test_image/). What happen if you try to use the post endpoint **before** training the model? This is bad as the user may not know if there are any trained models available.

What we can do is to use [lifespan events](https://fastapi.tiangolo.com/advanced/events/) to make sure there is at lease one model available when the application is running, it look something like this:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initial training model before appliction
    training()

    yield

    # Clean up the models and release resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
```

With lifespan events FastAPI guarantee the code **before yield** will be executed before the application starts taking requests, during the startup; and the code **after yield** will be executed after the application finishes handling requests, right before the shutdown.

In our case, we train the model for the first time during startup and clear al the models before shutdown.

Now when you run the application, you will see that after the message "Waiting for application startup." the model is under training, this is the initial training that is happening. After the application started, test if the `predict` endpoint works immediately.


### Exercise 3: re-training with new data

So far, we have been training the model using the default data provided in Keras. To simulate situation where there are new data that we can use to re-train and fine tune our model, we will repurpose the `train` endpoint to do some fine tuning on the existing model.

First, let's make some modification on the function `training`:

```python
def training(x_train = None, y_train = None):
    ...

    if (x_train is None) or (y_train is None):
        # Load the data and split it between train and test sets
        (x_train, y_train), _ = keras.datasets.mnist.load_data()

    ...

    if len(ml_models) > 0:
        model = ml_models[-1]
    else:
        model = keras.Sequential(
            ...
        )

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    ...
```

As you see, now we will take in the training data if it is provided and if it is provided we will use them instead of loading those from Keras. Next we will take the latest model and train on top rather than building a new one if there are already trained model available.

After that, we will remove the previous `train` endpoint and create a new one (with POST endpoint instead):

```python
@app.post("/train/")
async def train_model(files: list[UploadFile], background_tasks: BackgroundTasks):
    train_img = []
    labels = []

    for file in files:
        image = await file.read()
        image = Image.open(BytesIO(image)).convert('L')
        image = np.array(image)
        train_img.append(image)
        label = int(file.filename.split("_")[-1][0])
        labels.append(label)

    background_tasks.add_task(training, np.array(train_img), np.array(labels))
    return {"message": "Training model in the background"}
```

In this endpoint we will take multiple uploaded files. After processing each of them, the data is pass to the `training` function. The files that we are used for this training can be found inside [this zip file](data/train_img.zip), the digit after "_" in the file name is the target label of that image.

While you can try this endpoint with the interactive documentation, it is very tedious (and impossible) to use the given documentation UI to add in the files one by one for all 1000 images. We can build a simple UI in html with a new endpoint so we can upload multiple files at once:

```python
from fastapi.responses import HTMLResponse

@app.get("/upload/")
async def upload():
    content = """
<body>
<form action="/train/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
```

So after the application has been deployed, you can open the url http://127.0.0.1:8000/upload/ on your browser to upload the training files.

### Extra challenge

Now comes the extra challenge, could you create a new `retrain` endpoint that train the model again from starch?

### Exercise 4: avoid race condition

Finally, our app seems coming together, but we have a final problem. Remember we have talked about race condition briefly before? Now we should look at our app and see if we have any race condition issue.

Race condition is usually happening if memory is being access and modify with more than one process at the same time. Now, imagine if two training process is happening at the same time, they will both take the latest trained model and training on top of it, which means the weight of that model is modify by two training process at the same time. This is not good.

To avoid race condition, we have to change our design of the application. In stead of storing the trained model in memory, we will save a copy of it in the `models` folder after it is trained. And when ever the model is needed, the latest model saved will be loaded form the folder before used. This new design not just prevent race condition: the model saved in the memory being modify at the same time, but it also preserved the trained model incase the application needs to be restarted.

First, we will once again change the `training` function:

```python
def training(x_train = None, y_train = None):
    ...

    if len(ml_models) > 0:
        model = keras.saving.load_model(f"models/{ml_models[-1]}")
    else:
        ...

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    model_name = f"model_{datetime.now()}.keras"
    model.save(f"models/{model_name}")

    ml_models.append(model_name)
    num_train -= 1
```

Instead of storing the trained models, `ml_models` stores the file name of the model saved in the `models` folder. If there are trained models, the latest one will be loaded from the file. Because it is loaded from file, the file saved will not be modified while training is taking place.

We also have to make changes to the predict endpoint:

```python
@app.post("/predict/")
async def predict(file: UploadFile):
    ...
    # predict the result
    model = keras.saving.load_model(f"models/{ml_models[-1]}")
    result = model.predict(image).argmax(axis=-1)[0]
    return {"filename": file.filename,
            "result": str(result)}
```

Just like in `trianing` the latest model is loaded form the file.

### Extra challenge

Now, if the application has to be redeployed, the model will be training from scratch despite there are trained models saved in `models`. Could you add the functionality that during start up (remember the lifespan events?), the models saved will be loaded in the `ml_models` list (sorted in canonical order) and new models will be built and train only if there are no trained models available?

---

## Part 3 - Machine learning model in production

In the last part of this workshop, we will prepare our project for deployment. We will put our project in a Docker image and test running the container locally, so it will be ready to be put in the cloud service.

Assuming you have little to no Docker experience, this part of the workshop will go through the very basic of using a Docker container.

### Why using Docker?

Docker provide lightweight containers which allows us to deploy our application in an isolated and consistent environment. In [Docker Hub](https://hub.docker.com/) there are [pre-made official container image for Python](https://hub.docker.com/_/python) and is very useful in deploying Python application.

### Exercise 1: Creating images and running containers

To recreate our Python environment in a Docker image, one of the most important file is the `requirements.txt` which is already prepared [here](requirements.txt). If you are using other package management tools, you may also have other files. We will just use `pip` in the container to keep things simple.

Next, we will create our `Dockerfile`:

```
FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

CMD ["fastapi", "run", "main.py", "--port", "80"]
```

First, note that we are using Python 3.12 in deployment, it is also the Python version recommended in this workshop. Next, you will see many commands which are similar to commands that we used in our terminal:

- FROM: initialises a new build stage and sets the base image for subsequent instructions
- WORKDIR: sets the working directory
- COPY: copies new files or directories from your local directory and adds them to the filesystem of the image
- RUN: execute any commands to create a new layer on top of the current image
- CMD: command to be executed when running a container from an image

Note the difference between RUN and CMD, RUN is used to build new layers so usually is for installing new applications (or libraries) to set up the environment but CMD is used to execute new processes like running our FastAPI application. For details about the Dockerfile commands, please check their [official documentation](https://docs.docker.com/reference/dockerfile/).

Next, we need to build our image with the command:

```
docker build -t mnist_fastapi .
```

After that, we can start running the container:

```
docker run -d --name fastapi_app -p 80:80 mnist_fastapi
```

The option `-d` means that it is run in detach mode which run the container in background and leave your terminal still usable while it is running. `-p` is to publish the port 80 in the container to port 80 of the host. For details about `docker run` command, please see [documentation here](https://docs.docker.com/reference/cli/docker/container/run/).

If you are using Docker desktop, you can now monitor the container running using the Docker desktop dashboard. If you click `containers` you will see all your running containers. Click on the ID of a running container will let you see the "terminal" of the running container as if it is running locally. You will see the initial training in progress.

When the initial training finishes, you see that we have an error. It is because we need to save the trained model in a file and there is no hard-disk volume for the container to use.

You may now delete the container in the dashboard and move to the next step which we will fix this.

### Exercise 2: Mounting volumes

To let the running container to be able to save the trained model to our `models` folder, we need to bind mount a volume to the container. we can do so by adding the option `-v $(pwd):/code` in the `docker run` command:

```
docker run -d --name fastapi_app -p 80:80 -v $(pwd):/code mnist_fastapi
```

This will mount the current directory into the container `/code` directory, which is our working directory.

Now run the command again and wait for the container to start and the application to initialise. You can monitor the progress at Docker desktop. When the application is ready, you can open a browser and try out all the endpoints. For example, the root endpoint is http://127.0.0.1/.

### Extra challenge

So far we have a docker container running locally, although it is ready to be launch to a cloud service provider, we have not yet done so.

Normally you have to sign up an account with one of them and there are charges after the trial period. I decide to omit this part as I want to keep this workshop open source focused and not favour any cloud service providers. If you already have an account with a cloud service provider, or have a private server at home, feel free to try putting the docker container in the cloud.

I would also recommend looking into using [Docker compose](https://docs.docker.com/compose/) and  Infrastructure as Code (IaC) tools like [OpenTofu](https://opentofu.org/), which works very similar to [Terraform](https://developer.hashicorp.com/terraform/tutorials), when you are trying this part yourself.

---

## Support this workshop

This workshop is created by Cheuk and is open source for everyone to use (under MIT license). Please consider sponsoring Cheuk's work via [GitHub Sponsor](https://github.com/sponsors/Cheukting).
