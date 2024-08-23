from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Skinalyze API is running v23 aug."}

# Load models
try:
    acne_model = load_model("models/acne23augv1.h5")
    comedo_model = load_model("models/ComedoDetection_v2.h5")
    acne_level_model = load_model("models/AcneLVL_v2baru.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

@app.post("/Skinalyze-Predict/")
async def predictSkinalyze(file: UploadFile = File(...)):
    IMG_WIDTH, IMG_HEIGHT = 150, 150
    IMG_WIDTH_LVL, IMG_HEIGHT_LVL = 160, 160

    try:
        # Read and compress image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Compress image by resizing and reducing quality
        img = ImageOps.exif_transpose(img)  # Handle image orientation
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_acne = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_comedo = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_acne_lvl = img.resize((IMG_WIDTH_LVL, IMG_HEIGHT_LVL), Image.LANCZOS)

        # Prepare image arrays
        img_acne_array = np.expand_dims(np.array(img_acne) / 255.0, axis=0)
        img_comedo_array = np.expand_dims(np.array(img_comedo) / 255.0, axis=0)
        img_acne_lvl_array = np.expand_dims(np.array(img_acne_lvl) / 255.0, axis=0)

        # Predict Acne
        acne_classes = acne_model.predict(img_acne_array)
        acne_class_list = ['Acne', 'Clear']
        acne_prediction = acne_class_list[np.argmax(acne_classes[0])]

        # Predict Acne Level
        if acne_prediction == 'Clear':
            acne_level_prediction = 'Level 0'
        else:
            acne_level_classes = acne_level_model.predict(img_acne_lvl_array)
            acne_level_class_list = ['Level 1', 'Level 2', 'Level 3']
            acne_level_prediction = acne_level_class_list[np.argmax(acne_level_classes[0])]

        # Predict Comedo
        comedo_classes = comedo_model.predict(img_comedo_array)
        comedo_class_list = ['Clear', 'Comedo']
        comedo_prediction = comedo_class_list[np.argmax(comedo_classes[0])]

        return JSONResponse(content={
            "acne_prediction": acne_prediction,
            "acne_level_prediction": acne_level_prediction,
            "comedo_prediction": comedo_prediction
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
