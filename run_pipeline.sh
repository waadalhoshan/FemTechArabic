#!/bin/bash
# Run the complete FemTech analysis pipeline

echo "========================================"
echo "FemTech Arabic Reviews Analysis Pipeline"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Check if requirements are installed
echo "🔄 Checking dependencies..."
pip install -q -r requirements.txt

# Run pipeline
echo ""
echo "========================================="
echo "Starting analysis pipeline..."
echo "========================================="
echo ""

# Step 1
echo "STEP 1/7: Data Preprocessing"
python scripts/01_data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed. Exiting."
    exit 1
fi

# Step 2
echo ""
echo "STEP 2/7: Dialect Normalization"
python scripts/02_dialect_normalization.py
if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed. Exiting."
    exit 1
fi

# Step 3
echo ""
echo "STEP 3/7: Sentiment Analysis"
python scripts/03_sentiment_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed. Exiting."
    exit 1
fi

# Step 4
echo ""
echo "STEP 4/7: Topic Modeling"
python scripts/04_topic_modeling.py
if [ $? -ne 0 ]; then
    echo "❌ Step 4 failed. Exiting."
    exit 1
fi

# Step 5
echo ""
echo "STEP 5/7: Criteria Calculation"
python scripts/05_criteria_calculation.py
if [ $? -ne 0 ]; then
    echo "❌ Step 5 failed. Exiting."
    exit 1
fi

# Step 6
echo ""
echo "STEP 6/7: AHP Weighting"
python scripts/06_ahp_weighting.py
if [ $? -ne 0 ]; then
    echo "❌ Step 6 failed. Exiting."
    exit 1
fi

# Step 7
echo ""
echo "STEP 7/7: TOPSIS Ranking"
python scripts/07_topsis_ranking.py
if [ $? -ne 0 ]; then
    echo "❌ Step 7 failed. Exiting."
    exit 1
fi

echo ""
echo "========================================="
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================="
echo ""
echo "Results available in: output/"
echo "  • final_rankings.csv"
echo "  • criteria_scores.csv"
echo "  • topics.csv"
echo ""
