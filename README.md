# Replication Package: Prioritized Perspectives and Concerns in Arabic Reviews of FemTech Apps

This repository contains the complete replication package for the paper:

**"Prioritized Perspectives and Concerns in Arabic Reviews of FemTech Apps: A Mixed Method of Topic Modeling and Multi-Criteria Analysis"**

By: Waad Alhoshan  

## Overview

This replication package provides all code and documentation needed to reproduce the analysis of Arabic-language FemTech app reviews, including:
- Data preprocessing and cleaning
- Dialect normalization (dialectal Arabic → MSA)
- Sentiment analysis using transformer models
- Topic modeling with BERTopic
- Multi-criteria evaluation (AHP-TOPSIS)
- Prioritized ranking of user concerns and perspectives

## Repository Structure

```
replication_package/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration and parameters
├── LICENSE                            # MIT License
├── dataset/                           # Dataset folder
│   ├── README.md                      # Dataset documentation
│   └── reviews.csv                    # Raw reviews (I included mine but you can include your data of app reviews using the same template)
├── scripts/                           # Analysis pipeline scripts
│   ├── 01_data_preprocessing.py       # Clean and filter reviews
│   ├── 02_dialect_normalization.py    # Convert to MSA using GPT-4o
│   ├── 03_sentiment_analysis.py       # Sentiment classification
│   ├── 04_topic_modeling.py           # BERTopic clustering
│   ├── 05_criteria_calculation.py     # Calculate evaluation criteria
│   ├── 06_ahp_weighting.py            # AHP weights computation
│   └── 07_topsis_ranking.py           # TOPSIS ranking
├── output/                            # Generated results
│   ├── cleaned_reviews.csv
│   ├── msa_reviews.csv
│   ├── sentiment_results.csv
│   ├── topics.csv
│   ├── criteria_scores.csv
│   └── final_rankings.csv
├── figures/                           # Visualizations
└── docs/                              # Additional documentation
    └── METHODOLOGY.md                 # Detailed methodology notes
```

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- GPU recommended for transformer models (optional but faster)
- OpenAI API key (for dialect normalization)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/femtech-arabic-reviews.git
cd femtech-arabic-reviews

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Edit `config.py` and add your OpenAI API key:

```python
OPENAI_API_KEY = "your-api-key-here"
```

### 4. Prepare Dataset

Place your raw reviews CSV file in the `dataset/` folder as `reviews.csv`.

**Required columns**:
- `review_text`: The review content (in Arabic)
- `rating`: Star rating (1-5)
- `date`: Review submission date (YYYY-MM-DD)
- `app_version`: App version string
- `app_name`: Name of the app

### 5. Run the Pipeline

Execute scripts in order:

```bash
# Step 1: Clean and preprocess reviews
python scripts/01_data_preprocessing.py

# Step 2: Normalize dialectal Arabic to MSA
python scripts/02_dialect_normalization.py

# Step 3: Perform sentiment analysis
python scripts/03_sentiment_analysis.py

# Step 4: Extract topics using BERTopic
python scripts/04_topic_modeling.py

# Step 5: Calculate evaluation criteria
python scripts/05_criteria_calculation.py

# Step 6: Compute AHP weights
python scripts/06_ahp_weighting.py

# Step 7: Rank themes using TOPSIS
python scripts/07_topsis_ranking.py
```

Or run all steps at once:

```bash
bash run_pipeline.sh
```

## Expected Outputs

After running the pipeline, you'll find:

- **output/final_rankings.csv**: Ranked themes with TOPSIS scores
- **output/criteria_scores.csv**: Evaluation criteria values per theme
- **output/topics.csv**: All extracted topics with labels
- **figures/**: Generated visualizations (if matplotlib installed)

## Configuration Options

Edit `config.py` to customize:

- **Model parameters**: BERTopic settings, coherence thresholds
- **API settings**: OpenAI model, temperature, batch size
- **Criteria weights**: AHP weights (default from paper)
- **File paths**: Input/output locations

## Methodology

The analysis follows a 7-step pipeline:

1. **Data Preprocessing**: Remove noise, filter short reviews (<3 words)
2. **Dialect Normalization**: Convert dialectal Arabic to Modern Standard Arabic using GPT-4o
3. **Sentiment Analysis**: Classify sentiment using CAMeL Lab's Arabic BERT model
4. **Topic Modeling**: Extract topics using BERTopic (AraBERTv2 + HDBSCAN)
5. **Criteria Calculation**: Compute 4 evaluation criteria:
   - Topic Frequency (n_i / N)
   - User Importance Strength (rating × sentiment × confidence)
   - Review Recency (1 - d_i / D_max)
   - App Version Spread (v_i / V)
6. **AHP Weighting**: Apply Analytic Hierarchy Process weights
7. **TOPSIS Ranking**: Rank themes by similarity to ideal solution

For detailed methodology, see `docs/METHODOLOGY.md`.

## Citation

If you use this replication package, please cite: TO BE ADDED LATER 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

**Waad Alhoshan**  
Email: wmaboud@imamu.edu.sa  
Institution: Imam Mohammad ibn Saud Islamic University (IMSIU)

## Acknowledgments
- Best experts who worked with me and provided their insightful feedback! 
- CAMeL Lab for Arabic NLP models
- BERTopic framework
- OpenAI GPT-4o for dialect normalization
- All FemTech app users who contributed reviews

---

**Note**: This replication package is provided for academic and research purposes. Ensure you have appropriate permissions and comply with app store terms of service when collecting review data. However, our dataset of app reviews is available for testing AHP-TOPSIS pipeline, but you may also include yours.
