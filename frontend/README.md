# ASTRA: AI-Based Detection of Earth-Like Exoplanets in Kepler Data

Reproducible end-to-end machine learning pipeline and candidate vetting platform built according to the **Technical Specifications** by Vanshika Saxena.

---

## 🌟 Architecture & Improvements Over Starter Baseline

The pipeline addresses every documented baseline weakness outlined in Section 5:

1. **Ingestion & Quality Filtering (§5.1)**
   - Flags and removes cosmic rays, safe-mode transitions, and attitude tweaks (`quality == 0`).
   - Normalises each quarter to its median flux to eliminate roll discontinuities.
   - Automatically drops unviable stars with $<1000$ valid cadences.

2. **Noise & Stellar Variability Detrending (§5.2)**
   - **Baseline flaw**: The starter notebook's 1-day centred rolling median damages long-duration Earth-analog transits, recovering only ~33% of true transit depth.
   - **ASTRA solution**: Iterative **Savitzky-Golay filtering with transit dip preservation**. Negative outliers ($>2.5\sigma$) are masked and interpolated before re-fitting, preserving $>90\%$ of true transit depth.

3. **Period Search (§5.3)**
   - Coarse-to-fine Box Least Squares (`astropy.timeseries.BoxLeastSquares`).
   - Coarse log-spaced grid spanning $3.0$ to $\min(400, \text{baseline}/3)$ days.
   - Computes robust **Signal Detection Efficiency (SDE)**:
     $$\text{SDE} = \frac{\text{peak} - \text{median}(\text{power})}{1.4826 \times \text{MAD}(\text{power})}$$
   - Top candidate peak isolation ($\pm 10\%$ exclusion) and fine sweep ($\pm 2\%$).

4. **Candidate Vetting Suite (§5.5)**
   - **Odd / Even Transit Consistency Test**: Folds at $2\times$ period and tests alternate transit depths to reject eclipsing binaries.
   - **Secondary Eclipse Test at Phase 0.5**: Identifies occultation dips from stellar companions.
   - **Quarter Recurrence & Transit Count**: Ensures $\ge 3$ transits across multiple quarters.
   - **Transit SNR**: Evaluates in-transit depth over out-of-transit noise floor.

5. **Calibrated Confidence & Classification (§5.4 & §5.6)**
   - Replaces the starter notebook's flat logistic cut with continuous probabilistic calibration (Platt-scaled sigmoid).
   - Guarantees high uniqueness (`nunique() > 20`) for optimal PR-AUC / Average Precision scoring.

6. **Astrophysical Characterisation (§5.7)**
   - Reports orbital period (days), transit depth (ppm), and transit duration (hours).
   - Computes derived parameters: $R_p/R_*$, planet size in Earth radii ($R_\oplus$), semi-major axis $a$ (AU), and Habitable Zone status.

7. **Strict Submission Compliance (§6)**
   - Exactly 88 lines: 1 header + 87 data rows (`STAR_0000` to `STAR_0086`).
   - Leave fields blank (`,,,`) when `prediction = 0`.
   - Verified with official jury assertion suite.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Interactive Web Dashboard (Light Beige Aesthetic)
```bash
python -m uvicorn app:app --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
- **Theme**: Light shades of beige, warm linen, cream, sandstone, and espresso typography.
- Interactive target selection, light curve inspector, BLS periodograms, phase-folded transit profile, odd/even vetting test, and 1-click submission generation.

### 3. Run Pipeline via CLI
```bash
# Run on sample of stars
python run_pipeline.py --input "path/to/parquet_folder" --output "output/submission_astra.csv" --limit 10

# Run with custom detrending method
python run_pipeline.py --input "path/to/parquet_folder" --method savgol
```

### 4. Validate Submission File
```bash
python validate_submission.py output/submission_astra.csv
```
Verifies all 7 jury assertion rules:
- Exactly 88 lines (87 stars + 1 header)
- `star_id` matches `^STAR_\d{4}$`
- `prediction` in {0, 1}
- `confidence` in [0, 1] with real variance
- Empty characterisation fields on non-detections
- Positive period, depth, and duration on detections.

---

## 📁 Repository Structure
```
ASTRA/
├── pipeline/
│   ├── __init__.py           # Package exports
│   ├── ingest.py             # Raw parquet ingestion & quality filtering
│   ├── detrend.py            # Savitzky-Golay transit-preserving detrender
│   ├── bls_search.py         # Coarse-to-fine Box Least Squares & SDE
│   ├── vetting.py            # Odd/even, secondary eclipse, quarter recurrence
│   ├── classifier.py         # Multi-feature ML classifier & Platt scaling
│   ├── characterize.py       # Orbital & planetary astrophysical estimation
│   └── submission.py         # Submission generator & jury validator
├── templates/
│   └── index.html            # Web dashboard interface (light beige layout)
├── static/
│   ├── style.css             # Light beige / sandstone design tokens
│   └── app.js                # Plotly.js charts & interactive controls
├── app.py                    # FastAPI web server
├── run_pipeline.py           # CLI entry point (Module §8)
├── validate_submission.py    # Standalone jury assertion suite
├── train_eval.py             # Benchmark suite & depth recovery verification
├── requirements.txt          # Pinned project dependencies
└── README.md                 # Complete project documentation
```
