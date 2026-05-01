import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DIRECTORY MANAGEMENT & LOGGING
# ===================================
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# LOGGER
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # This ensures the text is pushed to the file in real-time
        self.terminal.flush()
        self.log.flush()

# Route terminal output to a RF-specific log file
sys.stdout = Logger("logs/rf_full_terminal_output.txt")

# LOAD THE PROCESSED DATA
# ============================================================
print("Loading data for Random Forest Model...")
data = np.load('traffic_flow_model_data/traffic_flow_model_input_sequences.npz')
X_train = data['train_inputs']
y_train = data['train_targets']
X_val = data['validation_inputs']
y_val = data['validation_targets']
X_test = data['test_inputs']
y_test = data['test_targets']

# Flatten the data
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print(f"Training shape after flatten: {X_train.shape}")

# BUILD THE ARCHITECTURE
# ===================================
model = RandomForestRegressor(
    n_estimators=100, # Number of decision trees (100 trees)
    max_depth=None,   # Trees can grow as deep as needed
    random_state=42,  # For reproducible results
    n_jobs=-1         # Use all CPU cores for faster training
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# EVALUATE ON THE TEST DATA
# ==============================================
print("\nEvaluating Random Forest...")
y_pred = model.predict(X_test) # Make predictions on test data

test_mae_scaled = mean_absolute_error(y_test, y_pred) # Average error magnitude
test_rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test MAE (compressed): {test_mae_scaled:.6f}")
print(f"Test RMSE (compressed): {test_rmse_scaled:.6f}")

# LOAD SCALER
with open('traffic_flow_model_data/traffic_flow_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Inverse transform to get real vehicle numbers
real_predictions = scaler.inverse_transform(y_pred.reshape(-1, 1))
real_actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Regression Metrics
mae = mean_absolute_error(real_actuals, real_predictions)
rmse = np.sqrt(mean_squared_error(real_actuals, real_predictions))
r2 = r2_score(real_actuals, real_predictions)

print("\n--- Random Forest Performance Report ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} vehicles")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} vehicles")
print(f"R-Squared Score: {r2:.4f}")
print("--------------------------------")

# Save the trained model
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/random_forest_model.pkl")

# Plotting the Learning Curve (Loss)
plt.figure(figsize=(12, 5))
plt.plot(real_actuals[:150], label='Actual Traffic', alpha=0.6)
plt.plot(real_predictions[:150], label='RF Prediction', linestyle='--')
plt.title('Traffic Flow: Actual vs Random Forest')
plt.xlabel('Hours')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)

plt.savefig("logs/rf_evaluation_plot.png")
plt.show()

# GENERATE TRAINING SUMMARY REPORT
# ===================================
print("\nGenerating Random Forest Summary...")

summary_text = f"""========================================
RANDOM FOREST MODEL SUMMARY
========================================

MODEL CONFIGURATION:
----------------------------------------
Number of Trees: 100
Max Depth: None (full growth)

DATA:
----------------------------------------
Input Features: 12 previous hours
Prediction: Next hour traffic

PERFORMANCE:
----------------------------------------
Test MAE:       {mae:.2f} vehicles
Test RMSE:      {rmse:.2f} vehicles
Test R-Squared: {r2:.4f}

NOTES:
----------------------------------------
Random Forest does not use epochs like neural networks.
It learns patterns from input features instead of temporal memory.

========================================
"""

with open("logs/rf_training_summary.txt", "w") as f:
    f.write(summary_text)

print("Summary saved to 'logs/rf_training_summary.txt'!")