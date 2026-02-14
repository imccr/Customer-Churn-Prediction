import pandas as pd
import numpy as np
import random

def generate_data(num_rows=5000):
    """
    Generates a synthetic Telecom Customer Churn dataset.
    """
    np.random.seed(42)
    random.seed(42)

    data = {
        'customerID': [f'{random.randint(1000,9999)}-{random.choice(["A","B","C"])}_{i}' for i in range(num_rows)],
        'gender': np.random.choice(['Male', 'Female'], num_rows),
        'SeniorCitizen': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], num_rows),
        'Dependents': np.random.choice(['Yes', 'No'], num_rows),
        'tenure': np.random.randint(1, 73, num_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], num_rows, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['No phone service', 'No', 'Yes'], num_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], num_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_rows),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], num_rows),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], num_rows),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, num_rows), 2)
    }

    df = pd.DataFrame(data)

    # Correlate Churn with Contract and Tenure for realistic patterns
    # Higher churn for Month-to-month contracts and lower tenure
    
    def simulate_churn(row):
        score = 0
        if row['Contract'] == 'Month-to-month':
            score += 0.4
        elif row['Contract'] == 'One year':
            score += 0.1
        
        if row['tenure'] < 12:
            score += 0.3
        elif row['tenure'] > 60:
            score -= 0.2
            
        if row['InternetService'] == 'Fiber optic':
            score += 0.1
            
        # Random noise
        score += np.random.normal(0, 0.2)
        
        if score > 0.5:
            return 'Yes'
        else:
            return 'No'

    df['Churn'] = df.apply(simulate_churn, axis=1)
    
    # Calculate TotalCharges (approximate)
    df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_data()
    output_path = "../data/telecom_churn.csv"
    try:
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except FileNotFoundError:
        # Fallback if running from root
        df.to_csv("data/telecom_churn.csv", index=False)
        print("Data saved to data/telecom_churn.csv")
