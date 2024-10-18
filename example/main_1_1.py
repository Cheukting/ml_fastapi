from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/users/{username}")
def read_user(username: str, q: Union[str, None] = None):
    return {"user": username, "q": q}
