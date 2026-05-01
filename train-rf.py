import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DIRECTORY MANAGEMENT & LOGGING
# ===================================
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# LOGGER
# ===================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("logs/rf_full_terminal_output.txt")

# LOAD THE PROCESSED DATA
# ============================================================
print("Loading data for Random Forest Model...")

data = np.load("traffic_flow_model_data/traffic_flow_model_input_sequences.npz")

X_train = data["train_inputs"]
y_train = data["train_targets"]
X_val = data["validation_inputs"]
y_val = data["validation_targets"]
X_test = data["test_inputs"]
y_test = data["test_targets"]

# Random Forest expects 2D tabular input:
# before flattening: (samples, 12, 1)
# after flattening:  (samples, 12)
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Training input shape after flattening: {X_train.shape}")
print(f"Validation input shape after flattening: {X_val.shape}")
print(f"Test input shape after flattening: {X_test.shape}")

# LOAD SCALER
# ============================================================
with open("traffic_flow_model_data/traffic_flow_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HELPER FUNCTIONS
# ============================================================
def calculate_scaled_metrics(actual_scaled, predicted_scaled):
    mae_scaled = mean_absolute_error(actual_scaled, predicted_scaled)
    rmse_scaled = np.sqrt(mean_squared_error(actual_scaled, predicted_scaled))
    return mae_scaled, rmse_scaled

def calculate_real_metrics(actual_scaled, predicted_scaled):
    real_actuals = scaler.inverse_transform(actual_scaled.reshape(-1, 1))
    real_predictions = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

    mae = mean_absolute_error(real_actuals, real_predictions)
    rmse = np.sqrt(mean_squared_error(real_actuals, real_predictions))
    r2 = r2_score(real_actuals, real_predictions)

    return real_actuals, real_predictions, mae, rmse, r2

# VALIDATION-BASED PARAMETER TESTING
# ============================================================
print("\nTesting Random Forest parameter settings using the validation set...")

parameter_options = [
    {
        "name": "RF_100_full_depth",
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "name": "RF_100_depth_20",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "name": "RF_100_depth_15",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "name": "RF_200_depth_20",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "name": "RF_100_depth_20_leaf_2",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2
    }
]

validation_results = []

for params in parameter_options:
    print(f"\nTraining candidate model: {params['name']}")

    candidate_model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

    candidate_model.fit(X_train, y_train)

    val_predictions = candidate_model.predict(X_val)

    val_mae_scaled, val_rmse_scaled = calculate_scaled_metrics(y_val, val_predictions)
    _, _, val_mae, val_rmse, val_r2 = calculate_real_metrics(y_val, val_predictions)

    validation_results.append({
        "params": params,
        "model": candidate_model,
        "val_mae_scaled": val_mae_scaled,
        "val_rmse_scaled": val_rmse_scaled,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

    print(f"Validation MAE (compressed scale):  {val_mae_scaled:.6f}")
    print(f"Validation RMSE (compressed scale): {val_rmse_scaled:.6f}")
    print(f"Validation MAE:                    {val_mae:.2f} vehicles")
    print(f"Validation RMSE:                   {val_rmse:.2f} vehicles")
    print(f"Validation R-Squared:              {val_r2:.4f}")

# Select the model with the lowest validation MAE.
best_result = min(validation_results, key=lambda result: result["val_mae"])
best_params = best_result["params"]
model = best_result["model"]

print("\n--- Best Random Forest Configuration ---")
print(f"Selected model: {best_params['name']}")
print(f"Number of trees: {best_params['n_estimators']}")
print(f"Max depth: {best_params['max_depth']}")
print(f"Min samples split: {best_params['min_samples_split']}")
print(f"Min samples leaf: {best_params['min_samples_leaf']}")
print(f"Best validation MAE: {best_result['val_mae']:.2f} vehicles")
print(f"Best validation RMSE: {best_result['val_rmse']:.2f} vehicles")
print(f"Best validation R-Squared: {best_result['val_r2']:.4f}")
print("----------------------------------------")

# FINAL TEST EVALUATION
# ============================================================
print("\nEvaluating selected Random Forest model on test data...")

test_predictions = model.predict(X_test)

test_mae_scaled, test_rmse_scaled = calculate_scaled_metrics(y_test, test_predictions)
real_actuals, real_predictions, test_mae, test_rmse, test_r2 = calculate_real_metrics(y_test, test_predictions)

print(f"Test MAE (compressed scale):  {test_mae_scaled:.6f}")
print(f"Test RMSE (compressed scale): {test_rmse_scaled:.6f}")

print("\n--- Random Forest Performance Report ---")
print(f"Mean Absolute Error (MAE): {test_mae:.2f} vehicles")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f} vehicles")
print(f"R-Squared Score: {test_r2:.4f}")
print("----------------------------------------")

# SAVE THE TRAINED MODEL
# ============================================================
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/random_forest_model.pkl")

# PLOT ACTUAL VS PREDICTED TRAFFIC
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(real_actuals[:150], label="Actual Traffic", alpha=0.6)
plt.plot(real_predictions[:150], label="Random Forest Prediction", linestyle="--")
plt.title("Traffic Flow: Actual vs Random Forest Predicted")
plt.xlabel("Hours")
plt.ylabel("Vehicles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/rf_evaluation_plot.png")
print("Evaluation plot saved to logs/rf_evaluation_plot.png")
plt.show()

# GENERATE TRAINING SUMMARY REPORT
# ============================================================
print("\nGenerating Random Forest Summary...")

validation_table_lines = []

for result in validation_results:
    params = result["params"]
    line = (
        f"{params['name']}: "
        f"Validation MAE = {result['val_mae']:.2f} vehicles, "
        f"Validation RMSE = {result['val_rmse']:.2f} vehicles, "
        f"Validation R-Squared = {result['val_r2']:.4f}"
    )
    validation_table_lines.append(line)

validation_table_text = "\n".join(validation_table_lines)

summary_text = f"""========================================
RANDOM FOREST MODEL SUMMARY
========================================

MODEL TYPE:
----------------------------------------
Random Forest Regressor

DATA:
----------------------------------------
Input Features: 12 previous hourly traffic-flow values
Prediction Target: Next-hour traffic flow
Training Shape: {X_train.shape}
Validation Shape: {X_val.shape}
Test Shape: {X_test.shape}

VALIDATION-BASED MODEL SELECTION:
----------------------------------------
{validation_table_text}

SELECTED CONFIGURATION:
----------------------------------------
Selected Model: {best_params['name']}
Number of Trees: {best_params['n_estimators']}
Max Depth: {best_params['max_depth']}
Min Samples Split: {best_params['min_samples_split']}
Min Samples Leaf: {best_params['min_samples_leaf']}

VALIDATION PERFORMANCE:
----------------------------------------
Validation MAE:       {best_result['val_mae']:.2f} vehicles
Validation RMSE:      {best_result['val_rmse']:.2f} vehicles
Validation R-Squared: {best_result['val_r2']:.4f}

REAL-WORLD TEST PERFORMANCE:
----------------------------------------
Test MAE:       {test_mae:.2f} vehicles
Test RMSE:      {test_rmse:.2f} vehicles
Test R-Squared: {test_r2:.4f}

NOTES:
----------------------------------------
Random Forest does not use epochs like LSTM or GRU.
Therefore, it does not have a neural-network learning curve.
The validation set is used to choose the best Random Forest parameter setting.
The test set is used only for final model evaluation.

Random Forest also does not keep temporal memory internally.
Instead, the previous 12 hours are flattened into 12 input features.

========================================
"""

with open("logs/rf_training_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Summary saved to logs/rf_training_summary.txt")