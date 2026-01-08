# Autojudge-ACM-23112026

# AutoJudge: Programming Problem Difficulty Prediction  
### Text-Based Machine Learning Approach

---

## Overview

AutoJudge is a machine learning system that predicts the difficulty of programming problems using **only textual problem statements**.

The system performs two tasks:
- **Difficulty Classification** → Easy / Medium / Hard  
- **Difficulty Regression** → Numeric difficulty score  

The project includes a complete pipeline covering data preprocessing, feature engineering, model training, evaluation, and a web-based interface for interactive predictions.

---

## Problem Statement

Given:
- Problem title  
- Problem description  
- Input format  
- Output format  

**Goal:**  
Automatically estimate the difficulty of a programming problem **without using solution code, test cases, or user performance data**.

This setup reflects real-world scenarios where difficulty must be estimated early, such as during problem creation or review.

---

## Dataset Used

The dataset consists of programming problems stored in **JSON format**.  
Each record represents one programming problem along with its difficulty information.

### Fields Used
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numeric difficulty)

Only textual data is used in this project.

---

## Approach

### 1. Text Preprocessing
- All text fields are merged into a single input
- Text is converted to lowercase
- Extra spaces and unnecessary characters are removed
- No advanced text transformations are applied

---

### 2. Feature Engineering

#### Text Features
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Maximum vocabulary size: 20,000
- Same vectorizer used for both classification and regression

#### Numeric Features
Additional features extracted from text:
- Total text length
- Title length
- Math symbol count
- Presence of constraints
- Multiple test case indicator
- Constraint density
- Algorithm keyword count

Numeric features are scaled before being used in regression.

---

## Models Used

### Difficulty Classification
- **Model:** XGBoost Classifier  
- **Input:** TF-IDF features  
- **Output:** Easy / Medium / Hard  

---

### Difficulty Regression
- **Model:** XGBoost Regressor  
- **Input:** TF-IDF + scaled numeric features  
- **Output:** Numeric difficulty score  

A fixed and stratified train–validation split is used for all experiments.

---

## Evaluation Metrics

### Classification
- **Accuracy:** 0.527  

### Regression
- **Mean Absolute Error (MAE):** 1.65  
- **Root Mean Squared Error (RMSE):** 1.99  

All results are reported on the validation dataset using the final saved models.

---

## Web Interface

A web application is developed using **Streamlit** for interactive predictions.

### Inputs
- Problem description  
- Input format  
- Output format  

### Outputs
- Predicted difficulty class  
- Predicted difficulty score  

The web interface follows the same preprocessing and feature extraction steps used during training.

---

## Steps to Run the Project Locally

### 1. Clone the Repository
```bash
git clone <your-github-repo-link>
cd AutoJudge
