import numpy as np
import keras
from keras import layers

ml_models = []
num_train = 0

def training():
    global num_train
    num_train += 1

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

    ml_models.append(model)
    num_train -= 1

from fastapi import UploadFile, BackgroundTasks, FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initial training model before appliction
    training()

    yield

    # Clean up the models and release resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

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
