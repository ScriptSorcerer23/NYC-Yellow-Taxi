# 🚕 NYC Yellow Taxi Fare Prediction App

An end-to-end Machine Learning project to predict taxi **fare amount** (regression) and **payment type** (classification), deployed using **Docker** and **AWS EC2**, with automated deployment via **GitHub Actions**.

---

## 🔍 Project Overview

This project uses the **NYC Yellow Taxi dataset** to perform:

- **Data Preprocessing** (PySpark using AWS EMR)
- **Model Training**:
  - Linear Regression (Fare Prediction)
  - Logistic Regression & Decision Tree (Payment Type)
  - KMeans Clustering
- **MLflow Logging** for experiment tracking
- **Deployment Stack**:
  - Frontend: Streamlit (Dockerized)
  - Backend: FastAPI 
  - CI/CD: GitHub Actions
  - Hosting: AWS EC2 instance (Ubuntu)

---

## 🚀 Features

- Predict **fare amount** using distance, tips, tolls, and surcharges
- Predict **payment type** based on ride info
- Cluster rides for insights using KMeans
- Interactive UI with custom styling
- Logged with **MLflow** and models stored on **AWS**

---

## 💻 Tech Stack

| Tool | Purpose |
|------|---------|
| PySpark | Big Data processing |
| Scikit-learn | ML models |
| MLflow | Experiment tracking |
| boto3 | S3 model upload |
| Streamlit | Frontend UI |
| Docker | Containerization |
| AWS EC2 | Cloud Hosting |
| GitHub Actions | CI/CD |

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/ScriptSorcerer23/NYC-Yellow-Taxi.git
cd NYC-Yellow-Taxi
docker build -t nyc-taxi-app .
docker run -d -p 80:8501 nyc-taxi-app
