# AI-ML Sentiment Analysis API

This project provides a **machine learning powered REST API for sentiment analysis** built with **FastAPI and scikit-learn**.
The API accepts text input and predicts whether the sentiment is **positive or negative**, returning the prediction along with a confidence score.

The system includes an end-to-end pipeline for:

* text preprocessing
* feature extraction using TF-IDF
* model training
* API deployment using FastAPI

---

# Project Overview

This project demonstrates how to **train a machine learning model and deploy it as a production-ready API**.

Main features:

* Text preprocessing pipeline
* TF-IDF feature extraction
* Logistic Regression sentiment classifier
* FastAPI REST endpoints
* Batch prediction support
* Health check endpoint

---

# Project Structure

```
AI-ML-Project
│
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading + prediction logic
│   ├── schemas.py       # Pydantic request/response models
│
├── static/
│   └── model/
│       ├── model.pickle
│       ├── tfidf_vectorizer.pickle
│       └── vocabulary.txt
│
├── train.py             # Training script
├── requirements.txt     # Project dependencies
└── README.md
```

---

# Setup Instructions

Follow the steps below to run the API locally.

## 1. Python Version

This project requires:

```
Python 3.9 or higher
```

---

## 2. Clone the Repository

```
git clone https://github.com/ChirathRasandun/AI-ML-Project.git
cd AI-ML-Project
```

---

## 3. Install Dependencies

Install all required packages using the `requirements.txt` file.

```
pip install -r requirements.txt
```

This installs the main libraries used in the project, including:

* FastAPI
* Uvicorn
* scikit-learn
* pandas
* nltk

---

## &#x20;

## 4. Train the Model

The sentiment model was originally developed and experimented with in the **Jupyter notebooks located in the **``** directory**. These notebooks were used for data exploration, preprocessing design, feature engineering, and model experimentation.

For reproducibility, the final training pipeline was converted into a standalone script. To train the model and generate the required artifacts, run:

```
python train.py

```

 

This script performs the following steps:

 

1. **Download and set up NLTK data** for preprocessing.
2. **Load the dataset**, remove duplicates, and handle null values.
3. **Preprocess the text**: lowercase conversion, link removal, punctuation and number removal, stopword filtering, and stemming.
4. **Build the vocabulary** by filtering tokens with frequency > 20.
5. **Split data into train and test sets** while maintaining class balance.
6. **Vectorize text using TF-IDF**.
7. **Train multiple models** (Logistic Regression, Random Forest, Naive Bayes, Decision Tree) and select the best performer.
8. **Tune the Logistic Regression model** using GridSearchCV.
9. **Evaluate the final model** on the test set and generate a classification report and confusion matrix.
10. **Save the trained model, TF-IDF vectorizer, vocabulary, and evaluation report** for inference

  

The following artifacts will be generated inside `static/model/`:

```
static/model/model.pickle
static/model/tfidf_vectorizer.pickle

```

These files are automatically loaded by the FastAPI service during startup.

  

---

## 5. Start the API Server

Run the FastAPI application using **Uvicorn**.

```
uvicorn app.main:app --reload
```

The API will start at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /health
```

Used to verify that the API and model are running.

   

Health check: curl -X GET "http://localhost:8000/health"

      

Example response:

```json
{ "status": "ok"}
```

---

## Predict Sentiment

```
POST /predict
```

Predict sentiment for a single text input.

### Request

```json
{
  "text": "This movie was amazing!"
}
```

### Response

```json
{
  "text": "This movie is absolutely amazing and wonderful!",
  "sentiment": "positive",
  "confidence": 0.9754
}
```

---

## Batch Prediction

```
POST /predict/batch
```

Predict sentiment for multiple texts in one request.

Example request:

```json
{
  "texts": [
    Absolutely fantastic! Best movie ever ,
    Waste of time and money,
    The acting was good but the plot was boring , 
    I would recommend this to everyone,  
    Terrible acting, poor direction  ]
}

```

  

---

# Example API Call (curl)

The following command sends a request to the `/predict` endpoint.

```
   
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"This movie is absolutely amazing and wonderful!\"}"
```

      

Example response:

```json
{
  "text": "This movie is absolutely amazing and wonderful!",
  "sentiment": "positive",
  "confidence": 0.9754
}

Example for Batch Prediction     
```

curl -X POST "http://localhost:8000/predict/batch" ^

  

  -H "Content-Type: application/json" ^

  

  -d "{\"texts\": [

  

    \"Absolutely fantastic! Best movie ever\",

  

    \"Waste of time and money\",

  

    \"The acting was good but the plot was boring\",

  

    \"I would recommend this to everyone\",

  

    \"Terrible acting, poor direction\"

  

  ]}" | python -m json.tool

      

Example Response :

"predictions": [

        {"text": "Absolutely fantastic! Best movie ever",

            "sentiment": "positive",

            "confidence": 0.9882},

        {"text": "Waste of time and money",

            "sentiment": "negative",

            "confidence": 0.9991},

{"text": "The acting was good but the plot was boring",

            "sentiment": "negative",

            "confidence": 0.9965},

        {"text": "I would recommend this to everyone",

            "sentiment": "positive",

            "confidence": 0.8434},

        {"text": "Terrible acting, poor direction",

            "sentiment": "negative",

            "confidence": 0.9996}],

    "count": 5

}

---

# Approach

This project uses **TF-IDF vectorization combined with a Logistic Regression classifier** to perform sentiment classification on text data. Logistic Regression was selected because it performs well on high-dimensional sparse text features and is computationally efficient for real-time inference. The preprocessing pipeline includes lowercasing, stopword removal, punctuation removal, and stemming to normalize the input text before vectorization.

With more time, I would experiment with **transformer-based models such as BERT**, which are capable of capturing deeper contextual relationships in text and often achieve higher accuracy in sentiment analysis tasks.

#


