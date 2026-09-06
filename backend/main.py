from fastapi import FastAPI, UploadFile, File
from PIL import Image
from gradcam import make_gradcam_heatmap
import tensorflow as tf
import numpy as np
import io
import base64


app = FastAPI(
    title="MedVision API",
    description="Chest X-ray Pneumonia Detection API",
    version="1.0.0"
)


# Load the trained MedVision model
model = tf.keras.models.load_model(
    "../models/medvision_densenet121.keras"
)

base_model = model.get_layer("densenet121")


@app.get("/")
def home():
    return {
        "message": "MedVision API is running!",
        "model": "DenseNet121"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image = image.convert("L")
    image = image.resize((150, 150))

    image_array = np.array(image) / 255.0

    input_image = np.expand_dims(
        image_array,
        axis=(0, -1)
    )

    probability = float(
        model.predict(
            input_image,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        predicted_class = "PNEUMONIA"
        confidence = probability
    else:
        predicted_class = "NORMAL"
        confidence = 1 - probability

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
        input_image,
        model,
        base_model
    )

    # Convert heatmap to PNG
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    buffer = io.BytesIO()
    heatmap_image.save(buffer, format="PNG")

    heatmap_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "pneumonia_probability": round(
            probability * 100,
            2
        ),
        "gradcam_image": heatmap_base64
    }