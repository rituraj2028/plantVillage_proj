 PlantVillage AI - Plant Disease Detection using Deep Learning

An end-to-end **Plant Disease Detection System** that uses a **Convolutional Neural Network (CNN)** to identify diseases in plant leaves. The application is deployed using **Google Cloud Functions**, stores the trained model in **Google Cloud Storage**, and provides a user-friendly **React** web interface for disease prediction.

---

##  Live Demo

**Frontend:** 
👉 https://rituraj2028.github.io/plantVillage_proj/

**GitHub Repository:** 
👉 https://github.com/rituraj2028/plantVillage_proj

---

#  Project Overview

Plant diseases can significantly reduce crop yield and quality. This project leverages Deep Learning to classify plant diseases from leaf images.

Users can upload an image of a plant leaf through a React web application, and the model predicts the disease along with a confidence score.

The application currently supports disease detection for:

-  Bell Pepper
-  Potato
-  Tomato

---

# Features

-  Plant disease prediction from leaf images
-  Custom CNN model built using TensorFlow/Keras
-  Google Cloud Functions for model inference
-  Google Cloud Storage for model hosting
-  Fast cloud-based predictions
-  Confidence score with every prediction
-  Responsive React frontend
-  REST API integration
-  Image upload support
-  Deployed on GitHub Pages

---

#  System Architecture

```text
                 React Frontend
               (GitHub Pages)
                       │
                       ▼
           Google Cloud Function
             (Prediction API)
                       │
                       ▼
        Google Cloud Storage Bucket
        (plantvillage_model.h5)
                       │
                       ▼
          TensorFlow CNN Model
                       │
                       ▼
     Disease Prediction + Confidence
```

---

#  CNN Architecture

```text
Input Image (256 × 256 × 3)
          │
          ▼
Resize & Rescale
          │
          ▼
Data Augmentation
          │
          ▼
Conv2D (32) + ReLU
          │
          ▼
MaxPooling2D
          │
          ▼
Conv2D (64) + ReLU
          │
          ▼
MaxPooling2D
          │
          ▼
Conv2D (64) + ReLU
          │
          ▼
MaxPooling2D
          │
          ▼
Conv2D (64) + ReLU
          │
          ▼
MaxPooling2D
          │
          ▼
Flatten
          │
          ▼
Dense (64) + ReLU
          │
          ▼
Dense (15) + Softmax
```

---

#  Supported Classes

### Bell Pepper

- Bacterial Spot
- Healthy

### Potato

- Early Blight
- Late Blight
- Healthy

### Tomato

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus
- Healthy

---

#  Tech Stack

## Frontend

- React 17
- Axios
- Material UI
- Material UI Dropzone

## Backend

- FastAPI (Local Development)

## Machine Learning

- TensorFlow 2.20
- Keras
- Custom CNN
- Data Augmentation
- MaxPooling
- Softmax Classifier

## Cloud

- Google Cloud Functions
- Google Cloud Storage

---

#  Project Structure

```text
plantVillage_proj/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── build/
│   ├── package.json
│   └── package-lock.json
│
├── gcp/
│   ├── main.py
│   └── requirements.txt
│
├── jupyter_notebook/plantvillage_project
│
├── models/
│   ├── saved_model_plantvillage/
│   ├── plantvillage_model.h5
│   ├── plantvillage_model.keras
│   └── saved_model.pb
│
└── README.md
```

---

#  Local Setup

## Clone Repository

```bash
git clone https://github.com/rituraj2028/plantVillage_proj.git

cd plantVillage_proj
```

---

## Backend

```bash
cd api

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

## Frontend

```bash
cd frontend

npm install

npm start
```

Frontend runs at:

```
http://localhost:3000
```

---

#  Google Cloud Deployment

The production application uses Google Cloud for scalable model inference.

Workflow:

1. User uploads a leaf image.
2. React frontend sends the image to the deployed Google Cloud Function.
3. Cloud Function downloads the trained model from Google Cloud Storage (only on the first request).
4. The model performs inference.
5. Predicted disease and confidence score are returned to the frontend.

---

### API Endpoint

## POST `/predict`

Uploads a plant leaf image for prediction.

## Response

```json
{
  "class": "Tomato_Early_blight",
  "confidence": 0.9874
}
```

---

#  Model Information

| Attribute | Value |
|-----------|-------|
| Model | Custom CNN |
| Framework | TensorFlow / Keras |
| Image Size | 256 × 256 |
| Number of Classes | 15 |
| Activation | ReLU |
| Output Activation | Softmax |
| Pooling | MaxPooling2D |
| Deployment | Google Cloud Functions |

---

#  Author

**Rituraj Singh Rathore**

GitHub: https://github.com/rituraj2028

---