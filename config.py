"""
Configuration file for FemTech Arabic Reviews Analysis
Edit this file to customize parameters, API keys, and file paths
"""

import os

# ============================================================================
# API CONFIGURATION
# ============================================================================

# OpenAI API key for dialect normalization (GPT-4o)
# IMPORTANT: Never commit your actual API key to GitHub!
# Use environment variables or a separate .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 150

# Batch size for API calls (to avoid rate limits)
NORMALIZATION_BATCH_SIZE = 100
NORMALIZATION_DELAY = 1.0  # seconds between batches

# ============================================================================
# FILE PATHS
# ============================================================================

# Input
INPUT_REVIEWS = "dataset/reviews.csv"

# Output
OUTPUT_DIR = "output"
CLEANED_REVIEWS = f"{OUTPUT_DIR}/cleaned_reviews.csv"
MSA_REVIEWS = f"{OUTPUT_DIR}/msa_reviews.csv"
SENTIMENT_RESULTS = f"{OUTPUT_DIR}/sentiment_results.csv"
TOPICS_FILE = f"{OUTPUT_DIR}/topics.csv"
CRITERIA_FILE = f"{OUTPUT_DIR}/criteria_scores.csv"
AHP_WEIGHTS_FILE = f"{OUTPUT_DIR}/ahp_weights.csv"
FINAL_RANKINGS = f"{OUTPUT_DIR}/final_rankings.csv"

# Figures
FIGURES_DIR = "figures"

# ============================================================================
# DATA PREPROCESSING PARAMETERS
# ============================================================================

# Minimum number of words for a review to be included
MIN_REVIEW_LENGTH = 3

# Remove reviews with only emojis or symbols
REMOVE_EMOJI_ONLY = True

# Remove excessive punctuation
REMOVE_EXCESSIVE_PUNCT = True

# ============================================================================
# SENTIMENT ANALYSIS PARAMETERS
# ============================================================================

# CAMeL Lab sentiment model
SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

# Confidence threshold (optional filtering)
MIN_SENTIMENT_CONFIDENCE = 0.0  # Set to 0 to keep all predictions

# ============================================================================
# TOPIC MODELING PARAMETERS
# ============================================================================

# AraBERT model for embeddings
ARABERT_MODEL = "aubmindlab/bert-base-arabertv2"

# BERTopic configuration
MIN_TOPIC_SIZE = 10  # Minimum reviews per topic
NR_TOPICS = 50  # Target number of topics after reduction
RANDOM_STATE = 42

# Topic coherence metric
COHERENCE_METRIC = "c_v"

# Reduction iterations to try
TOPIC_REDUCTION_RANGE = [350, 300, 250, 200, 150, 100, 80, 60, 50, 40, 30, 20, 10, 5]

# ============================================================================
# EVALUATION CRITERIA PARAMETERS
# ============================================================================

# Criteria calculation settings
NEUTRAL_RATING = 3.0  # Midpoint for user importance calculation

# Date format in input data
DATE_FORMAT = "%Y-%m-%d"

# ============================================================================
# AHP WEIGHTS (from paper)
# ============================================================================

# Final aggregated weights from the paper
AHP_WEIGHTS = {
    "Topic_Frequency": 0.3803,
    "User_Importance_Strength": 0.3441,
    "Review_Recency": 0.1117,
    "App_Version_Spread": 0.1639
}

# Expert pairwise comparison matrices (optional, for transparency)
# Saaty's scale: 1=Equal, 3=Moderate, 5=Strong, 7=Very strong, 9=Extreme
EXPERT_MATRICES = {
    "Expert1": {
        # Rows: Frequency, Importance, Recency, Version Spread
        "matrix": [
            [1, 2, 3, 2],
            [1/2, 1, 3, 2],
            [1/3, 1/3, 1, 1/2],
            [1/2, 1/2, 2, 1]
        ],
        "CR": 0.0170  # Consistency Ratio
    },
    "Expert2": {
        "matrix": [
            [1, 1, 5, 3],
            [1, 1, 5, 3],
            [1/5, 1/5, 1, 1/2],
            [1/3, 1/3, 2, 1]
        ],
        "CR": 0.0076
    },
    "Expert3": {
        "matrix": [
            [1, 1, 2, 3],
            [1, 1, 3, 2],
            [1/2, 1/3, 1, 1],
            [1/3, 1/2, 1, 1]
        ],
        "CR": 0.0780
    }
}

# ============================================================================
# TOPSIS PARAMETERS
# ============================================================================

# All criteria are benefit type (higher is better)
CRITERIA_TYPES = {
    "Topic_Frequency": "benefit",
    "User_Importance_Strength": "benefit",
    "Review_Recency": "benefit",
    "App_Version_Spread": "benefit"
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure style
FIGURE_DPI = 300
FIGURE_FORMAT = "png"  # png, svg, or pdf

# Color scheme
COLORS = {
    "Concern": "#D32F2F",
    "Perspective": "#1976D2"
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = f"{OUTPUT_DIR}/pipeline.log"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """Create output directories if they don't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✓ Created directories: {OUTPUT_DIR}, {FIGURES_DIR}")

def validate_config():
    """Validate configuration settings"""
    if OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  WARNING: OpenAI API key not set. Dialect normalization will fail.")
        print("   Set OPENAI_API_KEY environment variable or edit config.py")
    
    if not os.path.exists(INPUT_REVIEWS):
        print(f"⚠️  WARNING: Input file not found: {INPUT_REVIEWS}")
        print("   Place your reviews.csv in the dataset/ folder")
    
    # Check AHP weights sum to 1
    weights_sum = sum(AHP_WEIGHTS.values())
    if not (0.99 < weights_sum < 1.01):
        print(f"⚠️  WARNING: AHP weights sum to {weights_sum:.4f}, not 1.0")

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Model: {SENTIMENT_MODEL}")
    print(f"Target topics: {NR_TOPICS}")
    print(f"AHP Weights: {AHP_WEIGHTS}")
    validate_config()
