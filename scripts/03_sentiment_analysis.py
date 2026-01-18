"""
Script 03: Sentiment Analysis
Classifies sentiment of MSA reviews using CAMeL Lab's Arabic BERT model

Input: output/msa_reviews.csv
Output: output/sentiment_results.csv

Steps:
1. Load MSA-normalized reviews
2. Initialize CAMeL Lab sentiment model
3. Classify sentiment (positive/neutral/negative)
4. Extract confidence scores
5. Save results with sentiment labels
"""

import pandas as pd
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Transformers import
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("ERROR: transformers package not installed")
    print("Run: pip install transformers torch")
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def initialize_sentiment_model():
    """Initialize CAMeL Lab Arabic sentiment model"""
    print("Initializing sentiment analysis model...")
    print(f"   Model: {config.SENTIMENT_MODEL}")
    
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"   Device: {device_name}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL)
        
        # Create pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True
        )
        
        print("Model loaded successfully")
        return sentiment_pipeline
    
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None

def classify_sentiment(pipeline, text, max_length=512):
    """Classify sentiment for a single review"""
    try:
        # Truncate long texts
        if len(text) > max_length:
            text = text[:max_length]
        
        # Get predictions
        results = pipeline(text)[0]
        
        # Extract label and confidence
        # Results format: [{'label': 'positive', 'score': 0.95}, ...]
        best_result = max(results, key=lambda x: x['score'])
        
        label = best_result['label']
        confidence = best_result['score']
        
        # Get scores for all labels
        scores_dict = {r['label']: r['score'] for r in results}
        
        return {
            'sentiment_label': label,
            'sentiment_confidence': confidence,
            'positive_score': scores_dict.get('positive', 0.0),
            'neutral_score': scores_dict.get('neutral', 0.0),
            'negative_score': scores_dict.get('negative', 0.0)
        }
    
    except Exception as e:
        print(f"Error classifying text: {e}")
        return {
            'sentiment_label': 'neutral',
            'sentiment_confidence': 0.0,
            'positive_score': 0.0,
            'neutral_score': 0.0,
            'negative_score': 0.0
        }

def analyze_sentiment(input_file, output_file):
    """Main sentiment analysis pipeline"""
    
    print("="*60)
    print("STEP 03: SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load MSA reviews
    print(f"Loading MSA reviews from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} reviews")
    
    # Initialize model
    sentiment_pipeline = initialize_sentiment_model()
    if sentiment_pipeline is None:
        return False
    
    # Process reviews
    print(f"Analyzing sentiment for {len(df):,} reviews...")
    
    sentiment_results = []
    for text in tqdm(df['review_text'], desc="Classifying"):
        result = classify_sentiment(sentiment_pipeline, text)
        sentiment_results.append(result)
    
    # Add results to dataframe
    sentiment_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df, sentiment_df], axis=1)
    
    # Statistics
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total reviews:     {len(df):,}")
    print(f"\nSentiment distribution:")
    print(df['sentiment_label'].value_counts())
    print(f"\nSentiment percentages:")
    print(df['sentiment_label'].value_counts(normalize=True) * 100)
    print(f"\nAverage confidence scores:")
    for label in ['positive', 'neutral', 'negative']:
        mask = df['sentiment_label'] == label
        if mask.sum() > 0:
            avg_conf = df.loc[mask, 'sentiment_confidence'].mean()
            print(f"  {label:10s}: {avg_conf:.4f}")
    
    # Sentiment by rating
    print(f"\nSentiment distribution by rating:")
    sentiment_by_rating = pd.crosstab(
        df['rating'], 
        df['sentiment_label'], 
        normalize='index'
    ) * 100
    print(sentiment_by_rating.round(1))
    
    # Filter by confidence threshold (optional)
    if config.MIN_SENTIMENT_CONFIDENCE > 0:
        print(f"Filtering by confidence threshold: {config.MIN_SENTIMENT_CONFIDENCE}")
        before = len(df)
        df = df[df['sentiment_confidence'] >= config.MIN_SENTIMENT_CONFIDENCE]
        print(f"   Removed {before - len(df):,} low-confidence predictions")
        print(f"   Remaining: {len(df):,} reviews")
    
    # Save results
    print(f"Saving sentiment results to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df):,} reviews with sentiment labels")
    
    return True

if __name__ == "__main__":
    success = analyze_sentiment(config.MSA_REVIEWS, config.SENTIMENT_RESULTS)
    
    if success:
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 04_topic_modeling.py")
    else:
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS FAILED")
        print("="*60)
        sys.exit(1)
