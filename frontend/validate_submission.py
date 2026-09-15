"""
Validation Script for Kepler Exoplanet Detection Hackathon Submissions
Directly implements Section 6.3 & 6.4 and jury scoring script requirements.
"""
import sys
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.submission import validate_submission_dataframe, REQUIRED_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Validate hackathon submission CSV.")
    parser.add_argument("csv_path", nargs="?", default="output/submission_astra.csv", help="Path to submission CSV")
    args = parser.parse_args()

    print(f"\n========================================================")
    print(f" Validating Exoplanet Submission: {args.csv_path}")
    print(f"========================================================")

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[FAIL] to load CSV file: {e}")
        sys.exit(1)

    res = validate_submission_dataframe(df)

    print(f"* Total star rows    : {res['total_rows']} (Required: 87)")
    print(f"* Detected planets   : {res['detections']}")
    print(f"* Non-detections     : {res['non_detections']}")
    print(f"* Unique confidences : {res['unique_confidence']} (Spread: [{res['confidence_min']:.3f}, {res['confidence_max']:.3f}])")

    if res["valid"]:
        print("\n[SUCCESS] VALIDATION PASSED - File is 100% compliant with hackathon scoring requirements!")
        print("Ready for official submission.")
        sys.exit(0)
    else:
        print("\n[FAIL] VALIDATION ERRORS DETECTED:")
        for err in res["errors"]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
