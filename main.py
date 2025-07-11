from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
from keras.models import load_model
import json

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")  # Route GET à la racine
async def predict_image(file: UploadFile):
    buffer = await file.read()

    img = Image.open(io.BytesIO(buffer)).convert("RGB").resize((128, 128))
    arr = np.expand_dims(img, axis=0)

    models_path = os.path.join(os.path.dirname(__file__), ".\models\\")

    json_path = os.path.join(os.path.dirname(__file__), ".\class_names.json")

    latest_ver = f"model_v3.keras"

    print(f"chargement du fichier {latest_ver}")

    model = load_model(f"{models_path}{latest_ver}")

    res = model.predict(arr)

    with open(json_path, "r") as f:
        class_names = json.load(f)

        print(res, class_names)

        prediction = np.argmax(res)

        return {"prediction": f"{class_names[prediction]} ({np.max(res):.2f})"}