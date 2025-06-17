# 🚕 NYC Yellow Taxi Fare Prediction App

**An End-to-End Big Data Analytics Project** for predicting NYC Yellow Taxi fare amounts and payment types using scalable ML workflows and modern cloud deployment techniques.

---

## 🔍 Project Overview

This project is part of a **Big Data Analytics** initiative, built using the NYC Yellow Taxi dataset to demonstrate:

- ✅ Scalable data preprocessing with **PySpark** on **AWS EMR**
- ✅ Machine Learning with:
  - 🔹 **Linear Regression** (Fare Prediction)
  - 🔹 **Logistic Regression** & **Decision Tree** (Payment Type Classification)
  - 🔹 **KMeans Clustering** for trip segmentation
- ✅ MLflow experiment tracking and artifact storage on AWS S3
- ✅ Dockerized frontend (Streamlit)
- ✅ Automated deployment to AWS EC2 using **GitHub Actions (CI/CD)**

---

## 🚀 Features

- 🎯 Predict **Fare Amount** based on trip distance, tip, tolls, and surcharges
- 🧠 Predict **Payment Type** using ride metadata
- 📊 Visualize **KMeans clusters** for passenger trends
- 🌐 Custom **Streamlit UI** for interaction
- 🧪 Tracked with **MLflow**, stored on **AWS S3**
- 🔁 CI/CD pipeline from GitHub → EC2 deployment

---

## 💻 Tech Stack

| Tool/Tech         | Purpose                      |
|-------------------|------------------------------|
| PySpark           | Big Data preprocessing       |
| AWS EMR           | Distributed Spark processing |
| Scikit-learn      | Machine Learning models      |
| MLflow            | Experiment tracking          |
| boto3             | Model uploads to S3          |
| Streamlit         | Frontend UI                  |
| FastAPI           | Backend API (optional)       |
| Docker            | Containerization             |
| AWS EC2 (Ubuntu)  | Cloud deployment             |
| GitHub Actions    | CI/CD automation             |

---

## 🧪 How to Run Locally

git clone https://github.com/ScriptSorcerer23/NYC-Yellow-Taxi.git
cd NYC-Yellow-Taxi
docker build -t nyc-taxi-app .
docker run -d -p 80:8501 nyc-taxi-app

## 📁 Project Structure

NYC-Yellow-Taxi/
├── app.py                        # Streamlit frontend
├── main.py                       # Backend / utility logic
├── *.pkl                         # Trained ML models
├── *.png                         # Visualizations
├── Dockerfile                    # Docker build config
├── requirements.txt              # Python dependencies
├── .github/workflows/deploy.yml # CI/CD GitHub Actions workflow
└── README.md

## 🌩️ Deployment Details

- The Dockerized app is deployed to an **AWS EC2 Ubuntu instance**
- Using GitHub Actions CI/CD: any push to `main` branch redeploys the app automatically
- Docker handles building and running the container on the cloud

## 📦 Models & Artifacts

- Trained models:
  - `linear_regression_fare.pkl`
  - `logistic_regression_model.pkl`
  - `decision_tree_model.pkl`
  - `kmeans_model.pkl`
- Logged and versioned using MLflow
- Stored securely in AWS S3

## 📢 Author

Chaudhary Sumama Tahir — Final Year Data Science Student | GitHub: [@ScriptSorcerer23](https://github.com/ScriptSorcerer23)


