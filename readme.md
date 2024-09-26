# Length of Stay Prediction with MIMIC-III

This project propose the prediction of **Length of Stay (LoS)** utilizing a heterogeneous graph representation of the MIMIC-III dataset. Along with that, this project implements multiple machine learning models to predict the **Length of Stay (LoS)** for patients using the **MIMIC-III** clinical database. The models include:
- Artificial Neural Networks (ANN)
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

## Dataset

- **MIMIC-III** is a publicly available critical care database, and it contains de-identified health data of patients.
- Preprocessed and relevant features from patient data (e.g., demographics, diagnoses, vitals, etc.) are used for training and testing.

## Models

The following machine learning models are implemented:

1. **Artificial Neural Networks (ANN)**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **Logistic Regression**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/los-prediction.git
    cd los-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the MIMIC-III dataset:
    - Ensure the MIMIC-III data is preprocessed and stored in the `data/` directory.

## Usage

To train and evaluate models, run:
