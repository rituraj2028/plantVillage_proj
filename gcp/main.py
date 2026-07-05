from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import functions_framework

BUCKET_NAME = "plantvillage_ptbdc"
class_names = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy','Tomato__Target_Spot','Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']

model = None

def download_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob("models/plantvillage_model.h5")
    blob.download_to_filename("/tmp/plantvillage_model.h5")
    

@functions_framework.http
def predict(request):
    global model
    if model is None:
       download_model()
       model = tf.keras.models.load_model("/tmp/plantvillage_model.h5")

    image = request.files["file"]
    print("FIlename:",image.filename)
    image =np.array(Image.open(image).convert("RGB").resize((256,256)))
    
    img_array = tf.expand_dims(image,0)
    print("shape:",image.shape)
    print("min:",image.min())
    print("max:",image.max())
    print(image[0][0])
    print(image[10][10])

    predictions = model.predict(img_array)
    print("Predictions:",predictions)
    for i,p in enumerate(predictions[0]):
        print(class_names[i],float(p))
    print("Predicted index:",np.argmax(predictions[0]))

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(100 * (np.max(predictions[0])))

    return {"class": str(predicted_class),"confidence": round(confidence, 2)}


    