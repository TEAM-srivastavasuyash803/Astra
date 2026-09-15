import os
import glob
import re
import argparse
import time
import pandas as pd
from pipeline.engine import OptimizedExoplanetPipeline

EXPECTED_STARS = {f"STAR_{str(i).zfill(4)}" for i in range(87)} # STAR_0000 to STAR_0086[cite: 2]

def validate_submission_format(df: pd.DataFrame, expected_count: int = 87):
    """Executes strict assertion checks required by official submission format."""
    assert len(df) == expected_count, f"Submission must contain exactly {expected_count} rows, got {len(df)}"
    
    star_pattern = re.compile(r"^STAR_\d{4}$")
    for s_id in df['star_id']:
        assert star_pattern.match(str(s_id)), f"Invalid star_id pattern: {s_id}"

    assert df['star_id'].nunique() == len(df), "Duplicate star_id values detected"
    assert set(df['prediction'].unique()).issubset({0, 1}), "Prediction must be 0 or 1"
    assert df['confidence'].between(0.0, 1.0).all(), "Confidence must be within [0, 1]"

def write_reconciliation_log(found_stars: set, missing_stars: set, log_path: str):
    """Writes the mandatory auditability tie-in log[cite: 2]."""
    with open(log_path, 'w') as f:
        f.write(f"RECONCILIATION LOG\n")
        f.write(f"{len(found_stars)} files found vs. 87 expected[cite: 2].\n")
        if missing_stars:
            f.write(f"Missing star_id values ({len(missing_stars)}): {', '.join(sorted(missing_stars))}[cite: 2]\n")
        else:
            f.write("All 87 expected files are present.\n")
    print(f"Reconciliation log written to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Exoplanet Detection Pipeline")
    parser.add_argument("--input", required=True, help="Path to input directory containing STAR_*.parquet files")
    parser.add_argument("--output", required=True, help="Output submission CSV path")
    parser.add_argument("--sde-threshold", type=float, default=10.0, help="SDE threshold")
    parser.add_argument("--contingency", action="store_true", help="Enable partial-pack contingency to fill missing stars with 0s[cite: 2]")
    parser.add_argument("--sample", type=int, help="Run a small timed sample (e.g. 5) to extrapolate runtime[cite: 2]")
    args = parser.parse_args()

    pipeline = OptimizedExoplanetPipeline(sde_threshold=args.sde_threshold)
    parquet_files = sorted(glob.glob(os.path.join(args.input, "STAR_*.parquet")))

    # 1. Pre-flight count check & ID verification[cite: 2]
    found_stars = {os.path.basename(f).replace('.parquet', '') for f in parquet_files}
    missing_stars = EXPECTED_STARS - found_stars

    # Write the mandatory audit log alongside the CSV[cite: 2]
    log_path = args.output.replace('.csv', '_reconciliation.log')
    write_reconciliation_log(found_stars, missing_stars, log_path)

    # Abort loud condition[cite: 2]
    if missing_stars and not args.contingency:
        raise ValueError(f"ABORT: Pre-flight check failed! Found {len(found_stars)}/87 files. "
                         f"Missing {len(missing_stars)} IDs: {sorted(missing_stars)}. "
                         f"Use --contingency to bypass and zero-fill if deadline is near[cite: 2].")

    # Handle sampling for runtime extrapolation[cite: 2]
    if args.sample:
        parquet_files = parquet_files[:args.sample]
        print(f"Running timed sample of {args.sample} stars to extrapolate runtime...[cite: 2]")

    results = []
    start_time = time.time()

    for filepath in parquet_files:
        star_id = os.path.basename(filepath).replace('.parquet', '')
        df = pd.read_parquet(filepath)
        res = pipeline.process_star(star_id, df)
        
        # Ensure the backend strips the audit log for the final CSV submission
        if isinstance(res, dict) and 'result' in res:
            results.append(res['result'])
        else:
            results.append(res)

    # 2. Partial-pack contingency application[cite: 2]
    if missing_stars and args.contingency:
        print(f"Applying contingency: Zero-filling {len(missing_stars)} missing stars...[cite: 2]")
        for missing_id in missing_stars:
            results.append({
                'star_id': missing_id,
                'prediction': 0,
                'confidence': 0.0, # Must be 0 for missing files[cite: 2]
                'period': None,    # Characterisation fields left blank[cite: 2]
                'depth_ppm': None,
                'duration_hours': None
            })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values('star_id') # Re-sort to ensure STAR_0000 is first

    # Enforce strict column order
    columns = ['star_id', 'prediction', 'confidence', 'period', 'depth_ppm', 'duration_hours']
    out_df = out_df[columns]

    if not args.sample:
        validate_submission_format(out_df, expected_count=87)

    out_df.to_csv(args.output, index=False)
    
    elapsed = time.time() - start_time
    if args.sample:
        avg_time = elapsed / args.sample
        extrapolated_time = (avg_time * 87) / 60
        print(f"\nExtrapolated full-run time (87 stars): ~{extrapolated_time:.2f} minutes[cite: 2].")
    
    print(f"Generated submission file at: {args.output}")

if __name__ == "__main__":
    main()