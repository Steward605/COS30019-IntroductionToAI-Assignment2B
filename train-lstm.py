import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import sys
import os

# DIRECTORY MANAGEMENT + SAVE ALL OUTPUT/LOGS TO FILE
# ============================================================
# Automatically create the required folders if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

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

# Route terminal output to a LSTM-specific log file
sys.stdout = Logger("logs/lstm_full_terminal_output.txt")

# LOAD THE PROCESSED DATA
# ============================================================
print("Loading data for LSTM Model...")
data = np.load('traffic_flow_model_data/traffic_flow_model_input_sequences.npz')
X_train = data['train_inputs']
y_train = data['train_targets']
X_val = data['validation_inputs']
y_val = data['validation_targets']
X_test = data['test_inputs']
y_test = data['test_targets']
print(f"Training data shape: {X_train.shape}") # Should be (Samples, 12, 1)

# BUILD THE ARCHITECTURE
# ==============================================
# Sequential means we are stacking the layers one by one, top to bottom.
model = Sequential()

# The Input Layer: Tells the model to read the shape directly from the data
model.add(Input(shape=X_train.shape[1:]))

# The LSTM Layer: '64' is the number of LSTM cells inside this layer. Note: NUMBER IS TWEAKABLE (32,64,128) TO SEE WHAT PERFORMS BETTER LTR
model.add(LSTM(64))

# Dropout Layer: randomly ignores 20% of the previous layer's outputs during training to reduce overfitting
model.add(Dropout(0.2))

# The Output Layer: A standard Dense layer with '1' node because we want exactly 1 number as our final prediction (the traffic flow for the next hour).
model.add(Dense(1))

print("\n--- Model Architecture ---")
model.summary()

# COMPILE & TRAIN THE MODEL
# ==============================================
# optimizer='adam' is the best all-rounder for adjusting parameters for our situation.
# "According to Kingma et al., 2014, the method is "computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters"." - Tensorflow
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
print("\nStarting training...")

# Custom logger to force normal decimal notation when printing all important training metrics
class DecimalLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"Epoch {epoch + 1:02d} - "
            f"loss: {logs.get('loss', 0):.6f} - "
            f"rmse: {logs.get('rmse', 0):.6f} - "
            f"val_loss: {logs.get('val_loss', 0):.6f} - "
            f"val_rmse: {logs.get('val_rmse', 0):.6f}"
        )

# If model stops improving on the validation test for 5 epochs in a row, automatically stops training to not waste time.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Create a logger that saves the epoch results to a CSV file
csv_logger = CSVLogger('logs/lstm_training_log.csv', append=False)

# epochs=50, read the entire October dataset up to 50 times.
# batch_size=32, looks at 32 examples at a time before updating its gates.
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop, csv_logger, DecimalLogger("LSTM")]
)

# EVALUATE ON THE TEST DATA
# ==============================================
print("\nEvaluating LSTM on Test Data...")
test_mse, test_rmse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE (compressed scale):          {test_mse:.6f}")
print(f"Test RMSE / NRMSE (compressed scale): {test_rmse:.6f}")

# Save the trained model
model.save('models/lstm_traffic_model.keras')
print("Model saved to 'models/lstm_traffic_model.keras'")

# VISUAL EVALUATION & METRICS
# ==============================================
print("\nGenerating LSTM Evaluation Metrics...")

# Load the Scaler to convert 0-1 values back to real Car Counts
with open('traffic_flow_model_data/traffic_flow_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict the test data
raw_predictions = model.predict(X_test)

# Inverse transform to get real vehicle numbers
real_predictions = scaler.inverse_transform(raw_predictions)
real_actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(real_actuals, real_predictions))
r2 = r2_score(real_actuals, real_predictions)

# NRMSE uses the MinMaxScaler training range as the normalization factor.
normalization_factor = float(scaler.data_range_[0])
nrmse = rmse / normalization_factor
nrmse_percentage = nrmse * 100

print("\n--- LSTM Model Performance Report ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} vehicles")
print(f"Normalization Factor: {normalization_factor:.2f} vehicles")
print(f"Normalized RMSE (NRMSE): {nrmse:.6f} ({nrmse_percentage:.2f}%)")
print(f"R-Squared Score: {r2:.4f}")
print("------------------------------------")

# Plotting the Learning Curve (Loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (Error)')
plt.plot(history.history['val_loss'], label='Validation Loss (Error)')
plt.title('Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

# Plotting Actual vs Predicted Traffic (First 150 hours for clarity)
plt.subplot(1, 2, 2)
plt.plot(real_actuals[:150], label='Actual Traffic', color='blue', alpha=0.6)
plt.plot(real_predictions[:150], label='Predicted Traffic', color='red', alpha=0.8, linestyle='--')
plt.title('Traffic Flow: Actual vs. Predicted')
plt.xlabel('Hours')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/lstm_evaluation_plots.png')
print("Evaluation plots saved as 'logs/lstm_evaluation_plots.png'!")
plt.show()

# GENERATE TRAINING SUMMARY REPORT
# GENERATE TRAINING SUMMARY REPORT
# ===================================
print("\nGenerating Text Summary Report...")

# Extract the initial metrics (Epoch 1)
initial_loss = history.history['loss'][0] # MSE
initial_rmse = history.history['rmse'][0]
initial_val_loss = history.history['val_loss'][0]
initial_val_rmse = history.history['val_rmse'][0]

# Extract the final metrics (Last Epoch)
final_loss = history.history['loss'][-1] # MSE
final_rmse = history.history['rmse'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_rmse = history.history['val_rmse'][-1]

# Find the best validation RMSE and which epoch it happened on
best_val_rmse = min(history.history['val_rmse'])
best_epoch = history.history['val_rmse'].index(best_val_rmse) + 1

# AUTOMATED OVERFITTING ANALYSIS LOGIC
rmse_gap = final_val_rmse - final_rmse

if final_val_rmse > (final_rmse * 1.2): 
    overfitting_status = "WARNING: Potential Overfitting (Validation RMSE is significantly higher than training RMSE)."
elif rmse_gap > 0:
    overfitting_status = "HEALTHY: Generalizing well (Validation RMSE is slightly higher than training RMSE, which is normal)."
else:
    overfitting_status = "EXCELLENT: Validation RMSE is equal to or lower than training RMSE."

summary_text = f"""======================================================================
TRAINING SUMMARY - LEARNING CURVES (LSTM REGRESSION)
======================================================================

NOTE:
----------------------------------------------------------------------
The model was trained using compressed traffic-flow values, not the original vehicle counts. Because of this, the training and validation errors are shown on the compressed scale.

After testing, the predictions are converted back into real vehicle counts using the saved scaler. RMSE is reported in vehicles. NRMSE is calculated by dividing RMSE by the scaler's traffic-flow range.

INITIAL METRICS (Epoch 1):
----------------------------------------------------------------------
Training MSE (compressed scale):         {initial_loss:.4f}
Training RMSE (compressed scale):        {initial_rmse:.4f}

Validation MSE (compressed scale):       {initial_val_loss:.4f}
Validation RMSE (compressed scale):      {initial_val_rmse:.4f}

FINAL METRICS (Last Epoch):
----------------------------------------------------------------------
Training MSE (compressed scale):         {final_loss:.4f}
Training RMSE (compressed scale):        {final_rmse:.4f}

Validation MSE (compressed scale):       {final_val_loss:.4f}
Validation RMSE (compressed scale):      {final_val_rmse:.4f}

BEST METRICS DURING TRAINING:
----------------------------------------------------------------------
Best Validation RMSE (compressed scale): {best_val_rmse:.4f} (Achieved on Epoch {best_epoch})
Best Validation MSE (compressed scale):  {min(history.history['val_loss']):.4f}

IMPROVEMENT METRICS:
----------------------------------------------------------------------
MSE Reduction (compressed scale):        {initial_loss - final_loss:.4f}
RMSE Reduction (compressed scale):       {initial_rmse - final_rmse:.4f}

OVERFITTING ANALYSIS:
----------------------------------------------------------------------
Final RMSE Gap (Val - Train, compressed scale): {rmse_gap:.4f}
Generalization Status:               {overfitting_status}

REAL-WORLD TEST PERFORMANCE:
----------------------------------------------------------------------
Test RMSE:                 {rmse:.2f} vehicles
Normalization Factor:      {normalization_factor:.2f} vehicles
Test NRMSE:                {nrmse:.6f}
Test NRMSE Percentage:     {nrmse_percentage:.2f}%
Test R-Squared:            {r2:.4f}
======================================================================
"""

with open("logs/lstm_training_summary.txt", "w") as f:
    f.write(summary_text)

print("Summary saved to 'logs/lstm_training_summary.txt'!")