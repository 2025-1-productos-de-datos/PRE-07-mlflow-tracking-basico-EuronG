import mlflow
import pandas as pd

FILEPATH = "data/winequality-red.csv"

df = pd.read_csv(FILEPATH)
y = df["quality"]
x = df.drop(columns=["quality"])

logged_model = 'runs:/d5c5930471a14af28a2b562855dd6a73/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)