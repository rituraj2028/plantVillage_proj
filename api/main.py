from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from fastapi.middleware.cors import CORSMiddleware 


app =FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = tf.saved_model.load("../models/1/saved_model_plantvillage")
CLASS_NAME = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.get('/ping')
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
    ):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    
    infer = MODEL.signatures["serving_default"]
    predictions = infer(
        tf.constant(img_batch,dtype=tf.float32)
    )
    print(predictions)
    print(predictions.keys())
    predictions = predictions["output_0"].numpy()

    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }
print(MODEL.signatures)

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)    


