"""
convert_data.py — Kopal's data converter
Turns final_result.json into two CSVs the ML model expects:
  - exif_clean.csv   (EXIF features + aesthetic score)
  - proxy_clean.csv  (saturation + contrast proxies)

Run it like this:
  python convert_data.py
"""

import json
import pandas as pd

# ── 1. Load the JSON ──────────────────────────────────────────────────────────
print("Loading final_result.json...")
with open("final_result.json", "r") as f:
    data = json.load(f)

print(f"Found {len(data)} photos.")

# ── 2. Flatten into a DataFrame ───────────────────────────────────────────────
rows = []
for entry in data:
    row = {"image_id": entry["photo_id"]}
    row.update(entry["raw_features"])
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Columns found: {list(df.columns[:10])} ...")

# ── 3. Rename core EXIF columns to what the model expects ────────────────────
df = df.rename(columns={
    "ExposureTime_Num": "shutter_speed",
    "FNumber":          "aperture",
    "ISO":              "iso",
    "FocalLength":      "focal_length",
})

# ── 4. Fill missing EXIF values with column average ──────────────────────────
for col in ["shutter_speed", "aperture", "iso", "focal_length"]:
    missing = df[col].isna().sum()
    if missing > 0:
        df[col] = df[col].fillna(df[col].mean())
        print(f"  Filled {missing} missing values in '{col}' with column mean")

# ── 5. Add a dummy aesthetic score (replace later with real AVA scores) ───────
# If your JSON has a score field, change "aesthetic_score" below to match it
if "aesthetic_score" not in df.columns:
    print("  No aesthetic_score found — adding placeholder (5.0)")
    print("  ⚠️  Replace this with real AVA scores before final run!")
    df["aesthetic_score"] = 5.0

# ── 6. Split into exif CSV and proxy CSV ─────────────────────────────────────
exif_cols  = ["image_id", "shutter_speed", "aperture", "iso", "focal_length",
              "aesthetic_score"]

# also grab any CameraMake_ one-hot columns Chetan added
ohe_cols   = [c for c in df.columns if c.startswith("CameraMake_")]
exif_cols  = exif_cols + ohe_cols

proxy_cols = ["image_id", "Saturation_Proxy", "Contrast_Proxy"]

# only keep columns that actually exist
exif_cols  = [c for c in exif_cols  if c in df.columns]
proxy_cols = [c for c in proxy_cols if c in df.columns]

exif_df  = df[exif_cols].copy()
proxy_df = df[proxy_cols].copy()

# rename proxy columns to what the model expects
proxy_df = proxy_df.rename(columns={
    "Saturation_Proxy": "saturation_proxy",
    "Contrast_Proxy":   "contrast_proxy",
})

# ── 7. Save ───────────────────────────────────────────────────────────────────
exif_df.to_csv("exif_clean.csv",  index=False)
proxy_df.to_csv("proxy_clean.csv", index=False)

print(f"\n✅ Done!")
print(f"   exif_clean.csv  → {len(exif_df)} rows, {len(exif_df.columns)} columns")
print(f"   proxy_clean.csv → {len(proxy_df)} rows, {len(proxy_df.columns)} columns")
print(f"\nNow run your model with:")
print(f"   python kopal_ml_model.py --exif exif_clean.csv --proxy proxy_clean.csv")
