"""
Kepler Earth-Like Exoplanet Detection Pipeline
Developed for Vanshika Saxena's Hackathon Submission.
"""
from .ingest import load_kepler_data, clean_light_curve
from .detrend import detrend_light_curve, compare_detrending, iterative_savgol_detrend
from .bls_search import search_transit_bls, compute_sde
from .vetting import vet_transit_candidate, fold_light_curve
from .classifier import CandidateClassifier
from .characterize import characterize_candidate
from .submission import (
    generate_submission_file,
    validate_submission_dataframe,
    format_submission_row,
)

__all__ = [
    "load_kepler_data",
    "clean_light_curve",
    "detrend_light_curve",
    "compare_detrending",
    "iterative_savgol_detrend",
    "search_transit_bls",
    "compute_sde",
    "vet_transit_candidate",
    "fold_light_curve",
    "CandidateClassifier",
    "characterize_candidate",
    "generate_submission_file",
    "validate_submission_dataframe",
    "format_submission_row",
]
