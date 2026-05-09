# Food Expiry Tracker - Prediction Module

## Overview

This repository contains the Machine Learning prediction module used in the Food Expiry Tracker application. The module is responsible for predicting fruit freshness and quality using a trained Machine Learning model.

This repository only includes the prediction-related files such as trained models, encoders, and the main prediction script.

---

# Features

- Fruit quality prediction
- Food freshness detection
- Machine Learning based prediction
- Pre-trained ML model
- Fast prediction processing
- Lightweight prediction module

---

# Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Pickle (`.pkl` files)

---

# Repository Structure

```bash
food-expiry-tracker/
│
├── class_encoder.pkl
├── fruit_encoder.pkl
├── fruit_quality_model.pkl
├── main.py
├── requirements.txt
└── README.md
```

---

# File Description

## `fruit_quality_model.pkl`
Contains the trained Machine Learning model used for prediction.

## `fruit_encoder.pkl`
Encodes fruit categories into numerical values.

## `class_encoder.pkl`
Converts prediction outputs into readable labels.

## `main.py`
Main Python script responsible for loading the model and generating predictions.

## `requirements.txt`
Contains all required Python dependencies.

---

# Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# Run the Prediction Module

```bash
python main.py
```

---

# Prediction Output

The model predicts food quality categories such as:

- Fresh
- Good Quality
- Poor Quality
- Rotten / Expired

---

# Purpose of Repository

This repository serves as the prediction backend module for the main Food Expiry Tracker application.

---

# Future Improvements

- API integration
- Real-time prediction support
- Image-based food quality detection
- Deep Learning implementation
- Cloud deployment

---

