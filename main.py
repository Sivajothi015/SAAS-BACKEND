from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
import uvicorn
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import joblib
import pickle

app = FastAPI()

# Enable CORS for all origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient("MONGODB_URL")
db = client["userDB"]
activity_collection = db["user_activity"]

class Activity(BaseModel):
    name: str
    activity: str
    page: str
    timestamp: str  # ISO formatted string

@app.post("/activity")
async def log_activity(activity: Activity):
    # Check if the user has already visited this page
    existing_activity = activity_collection.find_one({
        "name": activity.name,
        "page": activity.page
    })

    if existing_activity:
        return {"message": "Activity already logged"}

    activity_data = activity.dict()
    activity_collection.insert_one(activity_data)
    return {"message": "Activity logged successfully"}

# SECTION 1: IMAGE PREDICTION

# Load CNN model and CSV
model_cnn = tf.keras.models.load_model('./myModel.h5')
df = pd.read_csv('./p4.csv')

# Preprocess uploaded image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_path = f"./{image.filename}"
    
    with open(image_path, "wb") as image_file:
        content = await image.read()
        image_file.write(content)
    
    processed_image = preprocess_image(image_path)
    prediction = model_cnn.predict(processed_image)

    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_class_names = [df.iloc[i]['Label'] for i in top3_indices]
    top3_scores = prediction[0][top3_indices]
    top3_percentages = top3_scores / np.sum(top3_scores) * 100

    response = {}
    for i in range(3):
        index = top3_indices[i]
        treatment = df.iloc[index]['Treatment']
        if pd.isna(treatment):
            treatment = "No treatment needed"

        response[f"prediction_{i+1}"] = {
            "class_name": top3_class_names[i],
            "confidence": f"{top3_percentages[i]:.2f}%",
            "example_picture": df.iloc[index]['Example Picture'],
            "description": df.iloc[index]['Description'],
            "prevention": df.iloc[index]['Prevention'],
            "treatment": treatment
        }
    
    os.remove(image_path)
    return JSONResponse(content=response)


# SECTION 2: USER SIGNUP - MongoDB

client = MongoClient("MONGODB_URL")
db = client["userDB"]
collection = db["users"]

class User(BaseModel):
    name: str
    phone: str

@app.post("/signup")
async def signup(user: User):
    existing_user = collection.find_one({"name": user.name, "phone": user.phone})
    
    if existing_user:
        return {"message": "User registered successfully"}
    
    if collection.find_one({"phone": user.phone}):
        raise HTTPException(status_code=400, detail="Phone number already used")
    if collection.find_one({"name": user.name}):
        raise HTTPException(status_code=400, detail="Name already used")

    collection.insert_one(user.dict())
    return {"message": "User registered successfully"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/get")
async def get():
    return JSONResponse(content={"message": "API is hitting perfectly"})


# SECTION 3: FERTILIZER PREDICTION

with open("classifier_fertilizer.pkl", "rb") as f:
    ferti_model = joblib.load(f)

with open("fertilizer.pkl", "rb") as f:
    ferti_labels = joblib.load(f)

@app.post("/predictfertilizer")
async def predict_fertilizer(
    temp: int = Form(...),
    humid: int = Form(...),
    mois: int = Form(...),
    soil: int = Form(...),
    crop: int = Form(...),
    nitro: int = Form(...),
    pota: int = Form(...),
    phos: int = Form(...)
):
    input_data = [temp, humid, mois, soil, crop, nitro, pota, phos]
    prediction_index = ferti_model.predict([input_data])[0]
    prediction_label = ferti_labels.classes_[prediction_index]

    return JSONResponse(content={"prediction": prediction_label})


# SECTION 4: CROP RECOMMENDATION

with open("classifier_crop.pkl", "rb") as f:
    crop_model = joblib.load(f)

crop_names = ['Apple','Banana','blackgram','chickpea','coconut','coffee',
     'cotton','grapes','jute','kidney beans','lentil','maize','mango',
     'moth beans','mung bean','muskmelon','orange','papaya','pigeonpeas',
     'pomegranate','Rice','Watermelon']

crop_df = pd.DataFrame({'label': crop_names, 'encoded': list(range(len(crop_names)))})
classes = crop_df.sort_values('encoded').set_index('encoded')

class CropInput(BaseModel):
    n: float
    p: float
    k: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predictcrop")
def predict_crop(data: CropInput):
    input_data = [[
        data.n, data.p, data.k, 
        data.temperature, data.humidity, 
        data.ph, data.rainfall
    ]]
    prediction = crop_model.predict(input_data)[0]
    crop_name = classes.loc[prediction].label.upper()
    return {"prediction": crop_name}


# MAIN ENTRY POINT

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
