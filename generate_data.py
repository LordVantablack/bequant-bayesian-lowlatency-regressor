import numpy as np
import pandas as pd
import os

# --- Settings ---
np.random.seed(42)  # <--- CRITICAL: Ensures reproducibility
NUM_SAMPLES = 100000  
NUM_FEATURES = 50     
NOISE = 0.5           

os.makedirs('data', exist_ok=True)

# 1. Generate X
X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)

# 2. Generate Betas
true_betas = np.random.randn(NUM_FEATURES)

# 3. Generate y
y = np.dot(X, true_betas) + np.random.normal(0, NOISE, NUM_SAMPLES)

# 4. Save DATA (X and y)
# We append y as the last column
data = np.column_stack((X, y))
pd.DataFrame(data).to_csv('data/dataset.csv', header=False, index=False)

# 5. Save TRUE BETAS (for validation)
pd.DataFrame(true_betas).to_csv('data/true_betas.csv', header=False, index=False)

print(f"Generated {NUM_SAMPLES} samples.")
print("Saved 'dataset.csv' (Input) and 'true_betas.csv' (Ground Truth).")