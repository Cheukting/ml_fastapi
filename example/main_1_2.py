import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/penguins.csv")
data = data.dropna()

le = preprocessing.LabelEncoder()
X = data[["bill_length_mm", "flipper_length_mm"]]
le.fit(data["species"])
y = le.transform(data["species"])
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)
clf.set_params().fit(X, y)

from fastapi import FastAPI

app = FastAPI()


@app.get("/predict/")
def predict(bill: float, flipper: float):
    param = {"bill_length": bill, "flipper_length": flipper}
    result = clf.predict([[bill, flipper]])
    return {
        "parameters": param,
        "result": le.inverse_transform(result)[0],
    }
