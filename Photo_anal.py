import pandas as pd
import numpy as np
import cv2
import urllib.request
import urllib.error
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import xgboost as xgb
import re
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration ---
DATASET_PATH = "photos.csv000"
CACHE_FILE = "proxy_cache.json"
FINAL_OUTPUT_FILE = "final_results.json"
NUM_ROWS_TO_PROCESS = None  # Set to 50 for testing

# --- 1. Helper Functions ---
def parse_exposure_time(value):
    if pd.isna(value) or value == "":
        return np.nan
    try:
        value = str(value).strip().lower()
        # Remove any letters (like 'sec' or 's')
        value = re.sub(r'[a-z]', '', value)
        if '/' in value:
            num, den = value.split('/', 1)
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return np.nan

def parse_fnumber(value):
    """Cleans strings like 'f/2.8', 'f2.8', '2.8f' into float 2.8"""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        value = str(value).strip().lower()
        # Remove 'f/' or 'f'
        value = value.replace('f/', '').replace('f', '')
        return float(value)
    except ValueError:
        return np.nan

def parse_generic_numeric(value):
    """Cleans strings like '50mm' or 'ISO 100' into floats"""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        value = str(value).strip()
        # Extract only digits and decimal points
        match = re.search(r'([0-9]+\.?[0-9]*)', value)
        if match:
            return float(match.group(1))
        return np.nan
    except ValueError:
        return np.nan

def extract_image_proxies_from_url(image_url, retries=3):
    """Downloads an image and extracts proxies, with built-in retry logic for network drops."""
    for attempt in range(retries):
        try:
            req = urllib.request.urlopen(image_url, timeout=10)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            
            if img is None:
                return None, None
                
            img_normalized = img.astype(np.float32) / 255.0

            # Proxies
            hsv_img = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2HSV)
            saturation_proxy = float(np.mean(hsv_img[:, :, 1]))
            
            gray_img = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
            contrast_proxy = float(np.std(gray_img))
            
            return saturation_proxy, contrast_proxy
            
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Network error on attempt {attempt + 1}/{retries} for {image_url}: {e}")
            time.sleep(2) # Wait before retrying
        except Exception as e:
            print(f"Processing error for {image_url}: {e}")
            break # Non-network error, no point in retrying
            
    return None, None

def calculate_vif(X_df):
    """Calculates Variance Inflation Factor (VIF) to prove multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    # VIF > 5 or 10 indicates high multicollinearity
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))]
    return vif_data

# --- Phase 1: Data Loading & Resilient Checkpointing ---
print(f"--- Starting Phase 1: Data Loading & Proxy Extraction (Limit: {NUM_ROWS_TO_PROCESS} rows) ---")
df = pd.read_csv(DATASET_PATH, sep='\t', nrows=NUM_ROWS_TO_PROCESS)

# Include 'exif_camera_make' for One-Hot Encoding
df.rename(columns={
    'exif_exposure_time': 'ExposureTime',
    'exif_aperture_value': 'FNumber',
    'exif_iso': 'ISO',
    'exif_focal_length': 'FocalLength',
    'exif_camera_make': 'CameraMake'
}, inplace=True)

# APPLY ALL CLEANERS
df['ExposureTime_Num'] = df['ExposureTime'].apply(parse_exposure_time)
df['FNumber'] = df['FNumber'].apply(parse_fnumber)
df['ISO'] = df['ISO'].apply(parse_generic_numeric)
df['FocalLength'] = df['FocalLength'].apply(parse_generic_numeric)
df['AestheticScore'] = np.log1p(df['stats_downloads'])

# Load existing progress if the script broke previously
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        proxy_cache = json.load(f)
    print(f"Loaded {len(proxy_cache)} previously processed images from cache.")
else:
    proxy_cache = {}

# Process only the URLs that aren't in the cache
for index, row in df.iterrows():
    photo_id = str(row['photo_id'])
    
    if photo_id in proxy_cache:
        continue # Skip already processed image
        
    url = row['photo_image_url']
    print(f"Processing {index + 1}/{len(df)}: ID {photo_id} ...", end=" ", flush=True)
    
    sat_proxy, cont_proxy = extract_image_proxies_from_url(url)
    
    if sat_proxy is not None and cont_proxy is not None:
        proxy_cache[photo_id] = {
            'Saturation_Proxy': sat_proxy,
            'Contrast_Proxy': cont_proxy
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(proxy_cache, f)
        print("Success.")
    else:
        proxy_cache[photo_id] = {'Saturation_Proxy': np.nan, 'Contrast_Proxy': np.nan}
        with open(CACHE_FILE, 'w') as f:
            json.dump(proxy_cache, f)
        print("Failed/Skipped.")

# Merge cached features back into the DataFrame
df['Saturation_Proxy'] = df['photo_id'].map(lambda pid: proxy_cache.get(str(pid), {}).get('Saturation_Proxy', np.nan))
df['Contrast_Proxy'] = df['photo_id'].map(lambda pid: proxy_cache.get(str(pid), {}).get('Contrast_Proxy', np.nan))

# --- Phase 2: Machine Learning, OHE, VIF & SHAP ---
print("\n--- Starting Phase 2: ML Preprocessing & Modeling ---")

# Drop rows where proxies couldn't be fetched
df_clean = df.dropna(subset=['Saturation_Proxy', 'Contrast_Proxy']).copy()

# ONE-HOT ENCODING: Convert categorical CameraMake into separate binary columns (0 or 1)
df_clean['CameraMake'] = df_clean['CameraMake'].fillna('Unknown')
df_encoded = pd.get_dummies(df_clean, columns=['CameraMake'], drop_first=True, dtype=float)

# Dynamically construct feature list to include numerical features + the new One-Hot columns
base_features = ['ExposureTime_Num', 'FNumber', 'ISO', 'FocalLength', 'Saturation_Proxy', 'Contrast_Proxy']
one_hot_features = [col for col in df_encoded.columns if col.startswith('CameraMake_')]
features = base_features + one_hot_features

X_raw = df_encoded[features]
y = df_encoded['AestheticScore']

# Handle missing EXIF data and Standardize
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=features, index=X_raw.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=features, index=X_raw.index)

# CALCULATE VIF (Variance Inflation Factor)
print("\n--- Variance Inflation Factor (VIF) Analysis ---")
print("Note: VIF > 5 indicates problematic multicollinearity between features.")
vif_results = calculate_vif(X_scaled)
print(vif_results.to_string(index=False))
print("------------------------------------------------\n")

# Train Model (Added max_depth to prevent RAM explosion and n_jobs=-1 for max speed)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, objective='reg:squarederror')
ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('xgb', xgb_model)])
ensemble_model.fit(X_scaled.values, y)

# Predict Scores
predictions = ensemble_model.predict(X_scaled.values)

# Calculate SHAP Values using high-speed TreeExplainer
print("Extracting trained models for SHAP analysis...")

# Extract the fully trained clones from the VotingRegressor
fitted_rf = ensemble_model.estimators_[0]
fitted_xgb = ensemble_model.estimators_[1]

explainer_rf = shap.TreeExplainer(fitted_rf)
explainer_xgb = shap.TreeExplainer(fitted_xgb)

# CHUNKING: Process SHAP in batches of 1000 to save RAM and show progress
print("Calculating SHAP values in chunks to prevent memory overload...")
chunk_size = 1000
shap_values_rf_list = []
shap_values_xgb_list = []

for i in range(0, len(X_scaled), chunk_size):
    end_idx = min(i + chunk_size, len(X_scaled))
    print(f" -> Processing SHAP for images {i} to {end_idx} (out of {len(X_scaled)})...")
    
    chunk = X_scaled.values[i:end_idx]
    
    # Calculate SHAP for the current chunk
    rf_chunk = explainer_rf.shap_values(chunk, check_additivity=False)
    xgb_chunk = explainer_xgb.shap_values(chunk, check_additivity=False)
    
    shap_values_rf_list.append(rf_chunk)
    shap_values_xgb_list.append(xgb_chunk)

# Combine all the chunks back together
shap_values_rf = np.vstack(shap_values_rf_list)
shap_values_xgb = np.vstack(shap_values_xgb_list)

# Average the SHAP values (since VotingRegressor is a 50/50 average)
shap_values_array = (shap_values_rf + shap_values_xgb) / 2.0
print("SHAP calculations complete!")
# --- 3. Construct Final JSON Output ---
final_output = []

for idx, (original_index, row) in enumerate(df_encoded.iterrows()):
    # Map SHAP values to their corresponding feature names dynamically
    shap_dict = {features[i]: float(shap_values_array[idx][i]) for i in range(len(features))}
    
    # Store Raw values dynamically to include One-Hot categorical flags
    raw_feature_dict = {
        feat: float(X_raw.loc[original_index, feat]) if not pd.isna(X_raw.loc[original_index, feat]) else None 
        for feat in features
    }
    
    # Construct Record
    record = {
        "photo_id": str(row['photo_id']),
        "photo_url": str(row['photo_image_url']),
        "raw_features": raw_feature_dict,
        "scores": {
            "actual_proxy_score": float(y.loc[original_index]),
            "predicted_score": float(predictions[idx])
        },
        "shap_values": shap_dict
    }
    final_output.append(record)

# Save the comprehensive JSON file
with open(FINAL_OUTPUT_FILE, 'w') as f:
    json.dump(final_output, f, indent=4)

print(f"\nPipeline Complete! Raw data and SHAP values successfully saved to: {FINAL_OUTPUT_FILE}")