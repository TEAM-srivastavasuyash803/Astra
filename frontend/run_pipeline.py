"""
End-to-End Pipeline Runner for Kepler Exoplanet Detection
Implements Section 5 and Section 8 (auditability, reproducible entry point).
"""
import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np

# Ensure Windows terminal doesn't crash on unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import (
    load_kepler_data,
    clean_light_curve,
    detrend_light_curve,
    search_transit_bls,
    vet_transit_candidate,
    CandidateClassifier,
    characterize_candidate,
    format_submission_row,
    generate_submission_file,
    validate_submission_dataframe,
)


def process_star(file_path, classifier, stellar_meta=None, method="savgol", verbose=False):
    """
    Executes full pipeline for a single star light curve.
    """
    star_name = os.path.splitext(os.path.basename(file_path))[0]
    star_id = star_name

    start_time = time.time()
    try:
        raw_df = load_kepler_data(file_path)
    except Exception as e:
        if verbose:
            print(f"[{star_id}] Ingestion failed: {e}")
        return format_submission_row(star_id, 0, 0.05), {"star_id": star_id, "status": "failed_ingestion", "error": str(e)}

    clean_dict = clean_light_curve(raw_df)
    if clean_dict is None:
        if verbose:
            print(f"[{star_id}] Discarded: <1000 cadences or unreadable data")
        return format_submission_row(star_id, 0, 0.05), {"star_id": star_id, "status": "insufficient_cadences"}

    # 1. Advanced Detrending (Iterative Savitzky-Golay with transit preservation)
    detrend_dict = detrend_light_curve(clean_dict, method=method)
    t = detrend_dict["time"]
    f = detrend_dict["flux"]
    q = detrend_dict["quarter"]

    # 2. Coarse-to-fine BLS search
    bls_res = search_transit_bls(t, f, n_coarse=15000, n_peaks=6, n_fine=400, verbose=verbose)
    
    # 3. Candidate Vetting
    vet_res = vet_transit_candidate(
        t, f, q,
        period=bls_res["period"],
        t0=bls_res["t0"],
        duration_hours=bls_res["duration_hours"],
        depth_ppm=bls_res["depth_ppm"]
    )

    # 4. Multi-Feature ML Classification & Confidence Calibration
    sp = stellar_meta.get(star_id, {}) if stellar_meta else {}
    prediction, confidence = classifier.predict(bls_res, vet_res, stellar_params=sp)

    # 5. Planetary Characterisation (only when prediction == 1)
    char_res = {}
    if prediction == 1:
        char_res = characterize_candidate(
            period=bls_res["period"],
            depth_ppm=bls_res["depth_ppm"],
            duration_hours=bls_res["duration_hours"],
            t0=bls_res["t0"],
            teff=sp.get("teff", 5778.0),
            logg=sp.get("logg", 4.438),
            radius=sp.get("radius", 1.0),
            kepmag=sp.get("kepmag", 12.0)
        )

    duration_sec = time.time() - start_time

    # Construct submission row adhering to Section 6 schema
    row = format_submission_row(
        star_id=star_id,
        prediction=prediction,
        confidence=confidence,
        period=char_res.get("period") if prediction == 1 else None,
        depth_ppm=char_res.get("depth_ppm") if prediction == 1 else None,
        duration_hours=char_res.get("duration_hours") if prediction == 1 else None,
    )

    audit_log = {
        "star_id": star_id,
        "prediction": prediction,
        "confidence": confidence,
        "sde": round(bls_res["sde"], 2),
        "period": char_res.get("period"),
        "depth_ppm": char_res.get("depth_ppm"),
        "duration_hours": char_res.get("duration_hours"),
        "rp_rs": char_res.get("rp_rs"),
        "planet_class": char_res.get("planet_class", "None"),
        "habitable_zone": char_res.get("habitable_zone", False),
        "odd_even_pass": vet_res["odd_even_pass"],
        "secondary_pass": vet_res["secondary_pass"],
        "transits": vet_res["transit_count"],
        "vetting_score": vet_res["vetting_score"],
        "reasons": "; ".join(vet_res["reasons"]) if vet_res["reasons"] else "Clean candidate",
        "runtime_s": round(duration_sec, 2),
    }

    return row, audit_log


def main():
    parser = argparse.ArgumentParser(description="Run full Kepler Exoplanet Detection Pipeline.")
    parser.add_argument("--input", default="C:/Users/VANSHIKA/Downloads/train_pack/train", help="Directory containing parquet light curves")
    parser.add_argument("--output", default="output/submission_astra.csv", help="Output submission CSV path")
    parser.add_argument("--labels", default=None, help="Path to labels CSV (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stars to process")
    parser.add_argument("--method", choices=["savgol", "baseline"], default="savgol", help="Detrending method")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostic output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" Kepler Earth-Like Exoplanet Detection Pipeline - ASTRA")
    print("=" * 60)
    print(f"* Input Directory   : {args.input}")
    print(f"* Output Submission : {args.output}")
    print(f"* Detrending Method : {args.method.upper()} (>90% depth recovery compliant)")
    
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not files:
        print(f"No .parquet files found in {args.input}")
        return

    if args.limit:
        files = files[:args.limit]

    print(f"* Found {len(files)} star light curves to process")

    stellar_meta = {}
    if args.labels and os.path.exists(args.labels):
        ldf = pd.read_csv(args.labels)
        for _, row in ldf.iterrows():
            kic_key = f"KIC_{int(row['kepid'])}"
            stellar_meta[kic_key] = row.to_dict()

    classifier = CandidateClassifier()

    rows = []
    audit_logs = []
    start_total = time.time()

    for idx, fpath in enumerate(files, 1):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        sub_row, audit = process_star(fpath, classifier, stellar_meta, method=args.method, verbose=args.verbose)
        rows.append(sub_row)
        audit_logs.append(audit)
        
        status_symbol = "[PLANET]" if sub_row["prediction"] == 1 else "[EMPTY] "
        print(f"[{idx:3d}/{len(files):3d}] {sid} | {status_symbol} Conf: {sub_row['confidence']:.3f} | SDE: {audit['sde']:5.1f} | {audit['runtime_s']:4.1f}s | {audit['reasons']}")

    total_time = time.time() - start_total
    avg_time = total_time / max(1, len(files))

    sub_df = generate_submission_file(rows, args.output)
    print(f"\nSaved submission file to: {args.output}")
    print(f"Total runtime: {total_time:.1f}s (Average: {avg_time:.2f}s/star)")

    val_report = validate_submission_dataframe(sub_df)
    print(f"Validation Status: {'PASSED' if val_report['valid'] else 'WARNINGS/ERRORS'}")
    if not val_report["valid"]:
        for err in val_report["errors"]:
            print(f"  - {err}")

    audit_path = os.path.splitext(args.output)[0] + "_audit.csv"
    pd.DataFrame(audit_logs).to_csv(audit_path, index=False)
    print(f"Audit report saved to: {audit_path}")


if __name__ == "__main__":
    main()
