# 🌾 Smart Agro Advisory System - Backend

The **Smart Agro Advisory System** backend is a FastAPI-powered service that offers smart agricultural insights, including crop recommendations, fertilizer suggestions, and plant disease predictions. This backend integrates machine learning and deep learning models trained using Python libraries to provide real-time, data-driven advisory support for farmers and agri-tech users.

---

## 🚀 Features

- 🔍 **Crop Recommendation** using **Random Forest** based on soil and weather parameters
- 🧪 **Fertilizer Recommendation** using **Reinforcement Learning**
- 🧠 **Plant Disease Prediction** using **Convolutional Neural Networks (CNN)**
- 👤 **User Signup** and tracking
- 📈 **API-first architecture** for easy frontend integration

---

## 🛠️ Tech Stack

- **FastAPI** – High-performance web framework for building APIs
- **Python Libraries**:
  - `scikit-learn` – Random Forest for crop recommendation
  - `TensorFlow` or `PyTorch` – CNN model for disease classification
  - `Pillow`, `OpenCV` – Image preprocessing
  - `pymongo` – MongoDB integration
- **MongoDB** – For storing user data and activity logs
- **Uvicorn** – ASGI server for running FastAPI

---


## 📡 API Endpoints

| Endpoint            | Method | Description                               |
|---------------------|--------|-------------------------------------------|
| `/signup`           | POST   | Register a new user                       |
| `/predictcrop`      | POST   | Predict the best crop using input data    |
| `/predictfertilizer`| POST   | Recommend fertilizer based on conditions  |
| `/predict`          | POST   | Predict plant disease from leaf image     |

---


  📦 Related Repositories
🔗 Frontend Repo: https://github.com/Sivajothi015/SAAS-FRONTEND



