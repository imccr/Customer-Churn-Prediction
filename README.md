# Customer Churn Prediction with Machine Learning 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

An end-to-end Machine Learning project to predict customer churn in the telecommunications industry. This project demonstrates the full ML lifecycle: from synthetic data generation to model training and deployment via an interactive Streamlit dashboard.  

## 🌟 Features

-   **Synthetic Data Generation**: Creates a realistic telco dataset using `numpy` and `pandas`.
-   **Machine Learning Pipeline**: 
    -   Data Preprocessing (Encoding, Scaling).
    -   Model Training (Random Forest Classifier). 
    -   Model Evaluation. 
-   **Interactive Dashboard**: A user-friendly web app built with Streamlit for real-time predictions.

## 🛠️ Tech Stack

-   **Language**: Python
-   **Libraries**: Pandas, NumPy, Scikit-learn, Streamlit, Joblib
-   **Algorithm**: Random Forest Classifier

## 📂 Project Structure

```bash
Customers-Churn-Prediction-ML/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data_generator.py   # Generates synthetic data
│   ├── model.py            # Trains and saves the model
├── models/                 # Saved model artifacts
├── app.py                  # Streamlit Dashboard
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Customers-Churn-Prediction-ML.git
cd Customers-Churn-Prediction-ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model

First, generate the synthetic dataset:
```bash
python src/data_generator.py
```

Then, train the machine learning model:
```bash
python src/model.py
```

### 4. Run the Dashboard

Launch the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Model Performance

The Random Forest model achieves high accuracy by analyzing key features such as:
-   Contract Type (Month-to-month vs. Yearly)
-   Tenure
-   Monthly Charges
-   Internet Service Type

## 🤝 Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)
