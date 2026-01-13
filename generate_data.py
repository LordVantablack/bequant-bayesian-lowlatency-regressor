import numpy as np
import pandas as pd
import os

# --- Settings ---
NUM_SAMPLES = 100000  
NUM_FEATURES = 50     
NOISE = 0.5           

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

print(f"Generating {NUM_SAMPLES} samples with {NUM_FEATURES} features...")

# 1. Generate Random X (Features)
X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)

# 2. Define "True" Coefficients (So we can verify later)
true_betas = np.random.randn(NUM_FEATURES)

# 3. Generate y (Target) = X * beta + Noise
y = np.dot(X, true_betas) + np.random.normal(0, NOISE, NUM_SAMPLES)

# 4. Combine into one DataFrame [Feature1, Feature2, ... Target]
data = np.column_stack((X, y))

# 5. Save to CSV (No Header, Pure Numbers)
df = pd.DataFrame(data)
csv_path = 'data/data.csv'
df.to_csv(csv_path, header=False, index=False)

print(f"Success! Data saved to {csv_path}")
print("First 5 True Betas (for checking C++ accuracy):")
print(true_betas[:5])