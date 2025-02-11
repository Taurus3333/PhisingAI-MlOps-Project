import os
import sys
import pandas as pd
import streamlit as st
import pymongo
import certifi
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL")
ca = certifi.where()

# Connect to MongoDB
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Train Model", "Predict"])

if menu == "Home":
    st.title("🤖 Welcome to AntiPhishAI app  💻🛜")

    st.write("Use the sidebar to navigate between different sections for TRAINING and PREDICTIONS")
    
    st.header("Project Description")
    st.write("""
    This project leverages MLOps tools to automate and streamline the machine learning pipeline, improving the overall efficiency. 
    It follows a modular coding approach and consists of five major components:

    - **Data Ingestion**: Extracts raw data from MongoDB, handling large datasets efficiently.
    - **Data Validation**: Ensures data quality by validating schema and detecting anomalies and drift, reducing data errors .
    - **Data Transformation**: Converts raw data into a structured format, thereby improving preprocessing speed . Utilizes **KNNImputer** with 3 neighbors to handle missing values.
    - **Model Training and Evaluation**: Trains an ensemble of classification models (**ANN, Logistic Regression, XGBoost, Decision Tree**), achieving an **f1-score of 0.991** on the training set and **0.977** on the test set.
    - **Model Pusher**: Deploys the trained model for inference, reducing deployment time considerably.
    - **Continuous Training**: Used **Airflow** for orchestrating automated retraining, reducing manual intervention in production environments.
    
    **Key Features:**
    - Comprehensive logging and custom exception handling for each component, reducing debugging time.
    - Uses **Config Entities** (schema definitions) and **Artifact Entities** (intermediate outputs) to ensure a structured workflow.
    - **CI/CD Integration**: Automates the pipeline using **GitHub Actions**, reducing manual deployment efforts.
    - **Experiment Tracking**: Utilizes **MLflow** to log and track different experiments enabling increased reproducibility 
    - **Model Serving**: Supports both **FastAPI** and **Streamlit** for deploying the trained model.
    - **Deployment**: 
      - The model is **Dockerized** and pushed to **ECR** . Finally it is deployed to **AWS EC2 instances**, thereby improving scalability .
      - Artifacts and logs are stored securely in **Amazon S3**, ensuring **99.9% availability**.
    
    **Performance Metrics:**
    - **Precision Score**: **0.989** (train), **0.971** (test)
    - **Recall Score**: **0.993** (train), **0.983** (test)
    - **F1 Score**: **0.991** (train), **0.977** (test)
    
    **Technologies Used:**
    - **Machine Learning Models**: Ensemble of classifiers (**ANN, Logistic Regression, XGBoost, Decision Tree**).
    - **MLOps Tools**: Airflow, MLflow, Docker, GitHub Actions.
    - **Cloud Services**: AWS EC2, AWS ECR, S3.
    """)
    
    st.write("**GitHub Repository:** [Click here](https://github.com/Taurus3333/Network-Security-MlOps.git)")

elif menu == "Train Model":
    st.title("Train Network Security Model")
    if st.button("Train Model"):
        try:
            train_pipeline = TrainingPipeline()
            train_pipeline.run_pipeline()
            st.success("Training is successful")
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Predict":
    st.title("Network Security Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.write(df.head())

            # Load Model
            preprocessor = load_object("final_model/preprocessor.pkl")
            final_model = load_object("final_model/model.pkl")
            network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

            # Make Predictions
            y_pred = network_model.predict(df)
            df['predicted_column'] = y_pred
            
            # Display Predictions
            st.write("### Predictions")
            st.write(df)
            
            # Option to Download the Predictions
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv_output, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")