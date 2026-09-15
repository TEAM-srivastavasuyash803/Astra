"""
Submission Generation & Strict Schema Validation Module
Complies with official hackathon submission specifications.
"""
import os
import re
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "star_id",
    "prediction",
    "confidence",
    "period",
    "depth_ppm",
    "duration_hours",
]


def format_submission_row(star_id, prediction, confidence, period=None, depth_ppm=None, duration_hours=None):
    """
    Formats a single submission row following the strict schema rules:
    When prediction = 0, characterisation fields must be None/empty (not 0, not -1, not NA).
    """
    row = {
        "star_id": str(star_id).strip(),
        "prediction": int(prediction),
        "confidence": round(float(confidence), 4),
    }

    if int(prediction) == 1:
        row["period"] = round(float(period), 5) if period is not None else None
        row["depth_ppm"] = round(float(depth_ppm), 1) if depth_ppm is not None else None
        row["duration_hours"] = round(float(duration_hours), 3) if duration_hours is not None else None
    else:
        # Strictly empty
        row["period"] = None
        row["depth_ppm"] = None
        row["duration_hours"] = None

    return row


def generate_submission_file(rows, output_path="submission_astra.csv"):
    """
    Generates CSV conforming to exact UTF-8, no index, 88 lines requirement.
    """
    df = pd.DataFrame(rows)
    df = df[REQUIRED_COLUMNS]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Write to CSV with empty string for NA values (producing ",,," as mandated)
    df.to_csv(output_path, index=False, na_rep="", encoding="utf-8")
    return df


def validate_submission_dataframe(df):
    """
    Runs the official jury assertion suite specified in evaluation criteria:
    Raises AssertionError or returns list of descriptive errors if non-compliant.
    """
    errors = []

    # 1. Header and columns
    if list(df.columns) != REQUIRED_COLUMNS:
        errors.append(f"Columns mismatch. Expected {REQUIRED_COLUMNS}, got {list(df.columns)}")

    # 2. Row count check
    if len(df) != 87:
        errors.append(f"Expected exactly 87 star rows, got {len(df)}")

    # 3. Duplicate star_id check
    if df["star_id"].nunique() != len(df):
        errors.append(f"Duplicate star_id detected ({df['star_id'].nunique()} unique out of {len(df)})")

    # 4. Star ID format regex
    bad_ids = df[~df["star_id"].astype(str).str.match(r"^STAR_\d{4}$")]["star_id"].tolist()
    if bad_ids:
        errors.append(f"Invalid star_id format in: {bad_ids[:5]}")

    # 5. Prediction values
    if not df["prediction"].isin([0, 1]).all():
        errors.append("prediction column must contain only 0 or 1")

    # 6. Confidence range and spread
    if not df["confidence"].between(0.0, 1.0).all():
        errors.append("confidence values must be bounded within [0, 1]")

    unique_conf = df["confidence"].nunique()
    if unique_conf <= 3:
        errors.append(f"Confidence has flat spread ({unique_conf} unique values). Will score poorly on PR-AUC.")

    # 7. Characterisation consistency
    pos = df[df["prediction"] == 1]
    for col in ["period", "depth_ppm", "duration_hours"]:
        if pos[col].isna().any():
            errors.append(f"Missing characterisation '{col}' in prediction=1 rows")

    if (pos["period"] <= 0).any():
        errors.append("Period must be strictly positive (> 0)")

    # 8. Non-detection rows should have no characterisation values
    neg = df[df["prediction"] == 0]
    for col in ["period", "depth_ppm", "duration_hours"]:
        non_empty = neg[col].dropna()
        if len(non_empty) > 0:
            errors.append(f"Prediction=0 rows have non-empty {col} values (must be blank)")

    is_valid = (len(errors) == 0)
    report = {
        "valid": is_valid,
        "errors": errors,
        "total_rows": len(df),
        "detections": int(df["prediction"].sum()) if "prediction" in df else 0,
        "non_detections": int((df["prediction"] == 0).sum()) if "prediction" in df else 0,
        "unique_confidence": unique_conf,
        "confidence_min": float(df["confidence"].min()) if len(df) else 0.0,
        "confidence_max": float(df["confidence"].max()) if len(df) else 0.0,
    }
    return report
