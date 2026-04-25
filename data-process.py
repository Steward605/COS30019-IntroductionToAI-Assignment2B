from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# SETTINGS
# ============================================================
RAW_SCATS_FILE = Path("Scats Data October 2006.xls")
OUTPUT_FOLDER = Path("traffic_flow_model_data")
INPUT_HISTORY_STEPS = 12 # use previous 12 hourly flow values
PREDICTION_HORIZON = 1 # predict 1 step ahead = next hour
TRAIN_SPLIT_RATIO = 0.70
VALIDATION_SPLIT_RATIO = 0.15
MIN_HOURLY_POINTS_PER_SITE = 48

# ============================================================
# STEP 1: READ RAW EXCEL DATA
# ============================================================
raw_sheet = pd.read_excel(RAW_SCATS_FILE,sheet_name="Data",header=None,engine="xlrd")
time_label_row = raw_sheet.iloc[0]
column_name_row = raw_sheet.iloc[1]
parsed_column_names = []
interval_code_to_time = {}

for column_index, raw_column_name in enumerate(column_name_row):
    if pd.isna(raw_column_name):
        parsed_column_names.append(f"unnamed_{column_index}")
        continue
    column_name = str(raw_column_name).strip()
    parsed_column_names.append(column_name)
    if column_name.startswith("V") and column_name[1:].isdigit() and len(column_name) == 3:
        interval_code_to_time[column_name] = pd.to_datetime(
            str(time_label_row[column_index])
        ).strftime("%H:%M:%S")
scats_daily_wide_df = raw_sheet.iloc[2:].copy()
scats_daily_wide_df.columns = parsed_column_names
scats_daily_wide_df = scats_daily_wide_df.reset_index(drop=True)

# ============================================================
# STEP 2: KEEP ONLY NEEDED COLUMNS AND CLEAN THEM
# ============================================================
scats_daily_wide_df = scats_daily_wide_df.rename(
    columns={
        "SCATS Number": "scats_site_number",
        "Date": "date"
    }
)
scats_daily_wide_df["scats_site_number"] = pd.to_numeric(scats_daily_wide_df["scats_site_number"],errors="coerce")
scats_daily_wide_df["date"] = pd.to_datetime(scats_daily_wide_df["date"],errors="coerce")
quarter_hour_flow_columns = [column_name for column_name in scats_daily_wide_df.columns if column_name in interval_code_to_time]
for column_name in quarter_hour_flow_columns:
    scats_daily_wide_df[column_name] = pd.to_numeric(scats_daily_wide_df[column_name],errors="coerce")
scats_daily_wide_df = scats_daily_wide_df.dropna(subset=["scats_site_number", "date"]).copy()
scats_daily_wide_df["scats_site_number"] = scats_daily_wide_df["scats_site_number"].astype(int)

# ============================================================
# STEP 3: CONVERT WIDE DATA TO LONG TIME-SERIES DATA
# ============================================================
traffic_flow_15min_df = scats_daily_wide_df.melt(id_vars=["scats_site_number", "date"],value_vars=quarter_hour_flow_columns,var_name="quarter_hour_interval_code",value_name="traffic_flow")
traffic_flow_15min_df["traffic_flow"] = pd.to_numeric(traffic_flow_15min_df["traffic_flow"],errors="coerce")
traffic_flow_15min_df = traffic_flow_15min_df.dropna(subset=["traffic_flow"]).copy()
traffic_flow_15min_df["time_text"] = traffic_flow_15min_df["quarter_hour_interval_code"].map(interval_code_to_time)
traffic_flow_15min_df["timestamp"] = pd.to_datetime(traffic_flow_15min_df["date"].dt.strftime("%Y-%m-%d")+ " "+ traffic_flow_15min_df["time_text"])
traffic_flow_15min_df = traffic_flow_15min_df[["scats_site_number", "timestamp", "traffic_flow"]].sort_values(["scats_site_number", "timestamp"]).reset_index(drop=True)

# ============================================================
# STEP 4: AGGREGATE 15-MINUTE FLOW INTO HOURLY FLOW
# ============================================================
traffic_flow_15min_df["timestamp"] = traffic_flow_15min_df["timestamp"].dt.floor("15min")
traffic_flow_15min_df["hour_timestamp"] = traffic_flow_15min_df["timestamp"].dt.floor("h")
processed_hourly_traffic_flow_df = (traffic_flow_15min_df.groupby(["scats_site_number", "hour_timestamp"], as_index=False)["traffic_flow"].sum().rename(columns={"hour_timestamp": "timestamp"}).sort_values(["scats_site_number", "timestamp"]).reset_index(drop=True))

# ============================================================
# STEP 5: DROP SITES WITH TOO LITTLE DATA
# ============================================================
hourly_point_counts_by_site = processed_hourly_traffic_flow_df.groupby("scats_site_number").size()
site_numbers_to_keep = hourly_point_counts_by_site[hourly_point_counts_by_site >= MIN_HOURLY_POINTS_PER_SITE].index
processed_hourly_traffic_flow_df = processed_hourly_traffic_flow_df[processed_hourly_traffic_flow_df["scats_site_number"].isin(site_numbers_to_keep)].copy()

# ============================================================
# STEP 6: SPLIT DATA INTO TRAIN / VALIDATION / TEST
# Plain words:
# - train = learn from this part
# - validation = check during development
# - test = final exam
# ============================================================
all_unique_timestamps = np.array(sorted(processed_hourly_traffic_flow_df["timestamp"].unique()))
number_of_unique_timestamps = len(all_unique_timestamps)
train_end_index = int(number_of_unique_timestamps * TRAIN_SPLIT_RATIO)
validation_end_index = int(number_of_unique_timestamps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))
train_end_timestamp = all_unique_timestamps[train_end_index - 1]
validation_end_timestamp = all_unique_timestamps[validation_end_index - 1]
hourly_traffic_flow_train_split_df = processed_hourly_traffic_flow_df[processed_hourly_traffic_flow_df["timestamp"] <= train_end_timestamp].copy()
hourly_traffic_flow_validation_split_df = processed_hourly_traffic_flow_df[(processed_hourly_traffic_flow_df["timestamp"] > train_end_timestamp)& (processed_hourly_traffic_flow_df["timestamp"] <= validation_end_timestamp)].copy()
hourly_traffic_flow_test_split_df = processed_hourly_traffic_flow_df[processed_hourly_traffic_flow_df["timestamp"] > validation_end_timestamp].copy()

# ============================================================
# STEP 7: SCALE FLOW VALUES USING TRAINING DATA ONLY
# Plain words:
# scaling = resizing the numbers to a standard range
# ============================================================
traffic_flow_scaler = MinMaxScaler()
hourly_traffic_flow_train_split_df[["scaled_traffic_flow"]] = traffic_flow_scaler.fit_transform(hourly_traffic_flow_train_split_df[["traffic_flow"]])
hourly_traffic_flow_validation_split_df[["scaled_traffic_flow"]] = traffic_flow_scaler.transform(hourly_traffic_flow_validation_split_df[["traffic_flow"]])
hourly_traffic_flow_test_split_df[["scaled_traffic_flow"]] = traffic_flow_scaler.transform(hourly_traffic_flow_test_split_df[["traffic_flow"]])

# ============================================================
# STEP 8: BUILD MODEL INPUT SEQUENCES
# Plain words:
# each sample = past traffic values -> next traffic value
# ============================================================
def create_model_input_sequences(flow_split_df, input_history_steps, prediction_horizon):
    model_inputs = []
    model_targets = []
    for _, site_flow_df in flow_split_df.groupby("scats_site_number"):
        site_flow_df = site_flow_df.sort_values("timestamp").reset_index(drop=True)
        scaled_flow_values = site_flow_df["scaled_traffic_flow"].to_numpy(dtype=np.float32)
        max_start_index = len(scaled_flow_values) - input_history_steps - prediction_horizon + 1
        if max_start_index <= 0:
            continue
        for start_index in range(max_start_index):
            end_index = start_index + input_history_steps
            target_index = end_index + prediction_horizon - 1
            model_inputs.append(scaled_flow_values[start_index:end_index].reshape(input_history_steps, 1))
            model_targets.append(scaled_flow_values[target_index])
    model_inputs = np.array(model_inputs, dtype=np.float32)
    model_targets = np.array(model_targets, dtype=np.float32)
    return model_inputs, model_targets

train_inputs, train_targets = create_model_input_sequences(hourly_traffic_flow_train_split_df,INPUT_HISTORY_STEPS,PREDICTION_HORIZON)
validation_inputs, validation_targets = create_model_input_sequences(hourly_traffic_flow_validation_split_df,INPUT_HISTORY_STEPS,PREDICTION_HORIZON)
test_inputs, test_targets = create_model_input_sequences(hourly_traffic_flow_test_split_df,INPUT_HISTORY_STEPS,PREDICTION_HORIZON)

# ============================================================
# STEP 9: SAVE OUTPUT FILES
# ============================================================
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
processed_hourly_traffic_flow_df.to_csv(OUTPUT_FOLDER / "processed_hourly_traffic_flow.csv",index=False)
hourly_traffic_flow_train_split_df.to_csv(OUTPUT_FOLDER / "hourly_traffic_flow_train_split.csv",index=False)
hourly_traffic_flow_validation_split_df.to_csv(OUTPUT_FOLDER / "hourly_traffic_flow_validation_split.csv",index=False)
hourly_traffic_flow_test_split_df.to_csv(OUTPUT_FOLDER / "hourly_traffic_flow_test_split.csv",index=False)
np.savez_compressed(OUTPUT_FOLDER / "traffic_flow_model_input_sequences.npz",train_inputs=train_inputs,train_targets=train_targets,validation_inputs=validation_inputs,validation_targets=validation_targets,test_inputs=test_inputs,test_targets=test_targets)
with open(OUTPUT_FOLDER / "traffic_flow_scaler.pkl", "wb") as scaler_file:
    pickle.dump(traffic_flow_scaler, scaler_file)

# ============================================================
# STEP 10: PRINT SUMMARY
# ============================================================
print("Data processing complete.")
print("Saved output folder:", OUTPUT_FOLDER.resolve())
print("Processed hourly rows:", len(processed_hourly_traffic_flow_df))
print("Training input shape:", train_inputs.shape)
print("Validation input shape:", validation_inputs.shape)
print("Test input shape:", test_inputs.shape)
print()
print("Saved files:")
print("- processed_hourly_traffic_flow.csv")
print("- hourly_traffic_flow_train_split.csv")
print("- hourly_traffic_flow_validation_split.csv")
print("- hourly_traffic_flow_test_split.csv")
print("- traffic_flow_model_input_sequences.npz")
print("- traffic_flow_scaler.pkl")