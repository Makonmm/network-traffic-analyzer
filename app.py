from fastapi import FastAPI, UploadFile, File
import pandas as pd
import torch
import joblib
from model import MainNetwork
from io import StringIO
import torch.nn.functional as F

app = FastAPI()

# Carrega a pipeline
preprocessor = joblib.load("preprocessor.pkl")


def load_model():
    model = MainNetwork()
    model.load_state_dict(torch.load(
        "model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()


def data_process(file: UploadFile):
    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    transformed_data = preprocessor.transform(df)
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()
    tensor = torch.tensor(transformed_data, dtype=torch.float32)
    return tensor


@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    try:
        tensor = data_process(file)
        with torch.no_grad():
            outputs = model(tensor)                      # shape: [N, 1]
            probs_1 = torch.sigmoid(outputs).squeeze()   # shape: [N]

            predictions = [
                {"prob_0": float(1 - p), "prob_1": float(p)}
                for p in probs_1
            ]

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}
