import pandas as pd
import numpy as np

# Random seed for reproducibility
np.random.seed(42)
n_samples = 300

data = {
    'fixed_acidity': np.random.uniform(4.0, 15.0, n_samples),
    'volatile_acidity': np.random.uniform(0.1, 1.5, n_samples),
    'citric_acid': np.random.uniform(0.0, 1.0, n_samples),
    'residual_sugar': np.random.uniform(0.9, 15.0, n_samples),
    'chlorides': np.random.uniform(0.01, 0.2, n_samples),
    'free_sulfur_dioxide': np.random.uniform(1, 70, n_samples),
    'total_sulfur_dioxide': np.random.uniform(6, 289, n_samples),
    'density': np.random.uniform(0.990, 1.003, n_samples),
    'pH': np.random.uniform(2.7, 4.0, n_samples),
    'sulphates': np.random.uniform(0.3, 2.0, n_samples),
    'alcohol': np.random.uniform(8.0, 15.0, n_samples),
}

df = pd.DataFrame(data)

# Quality logic based on simple threshold formula
df['quality'] = (
    (df['alcohol'] * 0.4) - (df['volatile_acidity'] * 2.5) + (df['sulphates'] * 1.2)
)
df['quality'] = np.where(df['quality'] > 3.5, 1, 0) # 1 = Good Quality, 0 = Bad Quality

df.to_csv('wine_data.csv', index=False)
print("wine_data.csv successfully generated!")