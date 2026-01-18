"""
Script 05: Criteria Calculation
Calculates the four evaluation criteria for each topic

Input: output/topics.csv
Output: output/criteria_scores.csv

Criteria:
1. Topic Frequency: n_i / N
2. User Importance Strength: |rating - 3| × sentiment × confidence
3. Review Recency: 1 - (d_i / D_max)
4. App Version Spread: v_i / V
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def calculate_topic_frequency(df):
    """
    Calculate topic frequency: n_i / N
    n_i = number of reviews for topic i
    N = total number of reviews
    """
    print("Calculating Topic Frequency...")
    
    total_reviews = len(df)
    topic_counts = df['topic'].value_counts()
    
    frequencies = {}
    for topic_id in topic_counts.index:
        if topic_id != -1:  # Exclude outliers
            frequencies[topic_id] = topic_counts[topic_id] / total_reviews
    
    print(f"Calculated frequencies for {len(frequencies)} topics")
    return frequencies

def calculate_user_importance(df):
    """
    Calculate user importance strength:
    |rating - 3| × sentiment_polarity × confidence
    
    sentiment_polarity: positive=+1, neutral=0, negative=-1
    """
    print("Calculating User Importance Strength...")
    
    # Map sentiment to polarity
    sentiment_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    
    df['sentiment_polarity'] = df['sentiment_label'].map(sentiment_map)
    
    # Calculate importance score for each review
    df['importance_score'] = (
        abs(df['rating'] - config.NEUTRAL_RATING) * 
        df['sentiment_polarity'] * 
        df['sentiment_confidence']
    )
    
    # Calculate average importance per topic
    importance_scores = {}
    for topic_id in df['topic'].unique():
        if topic_id != -1:
            topic_reviews = df[df['topic'] == topic_id]
            avg_importance = topic_reviews['importance_score'].mean()
            importance_scores[topic_id] = avg_importance
    
    print(f"Calculated importance scores for {len(importance_scores)} topics")
    return importance_scores

def calculate_review_recency(df):
    """
    Calculate review recency: 1 - (d_i / D_max)
    d_i = days since review
    D_max = age of oldest review in dataset
    """
    print("Calculating Review Recency...")
    
    # Convert date to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate days since each review
    max_date = df['date'].max()
    min_date = df['date'].min()
    D_max = (max_date - min_date).days
    
    df['days_since'] = (max_date - df['date']).dt.days
    df['recency_score'] = 1 - (df['days_since'] / D_max)
    
    # Calculate average recency per topic
    recency_scores = {}
    for topic_id in df['topic'].unique():
        if topic_id != -1:
            topic_reviews = df[df['topic'] == topic_id]
            avg_recency = topic_reviews['recency_score'].mean()
            recency_scores[topic_id] = avg_recency
    
    print(f"Calculated recency scores for {len(recency_scores)} topics")
    print(f"Date range: {min_date.date()} to {max_date.date()} ({D_max} days)")
    return recency_scores

def calculate_version_spread(df):
    """
    Calculate app version spread: v_i / V
    v_i = number of unique app versions for topic i
    V = total number of unique app versions
    """
    print("Calculating App Version Spread...")
    
    total_versions = df['app_version'].nunique()
    
    version_spreads = {}
    for topic_id in df['topic'].unique():
        if topic_id != -1:
            topic_reviews = df[df['topic'] == topic_id]
            unique_versions = topic_reviews['app_version'].nunique()
            version_spreads[topic_id] = unique_versions / total_versions
    
    print(f"Calculated version spread for {len(version_spreads)} topics")
    print(f"   Total unique versions: {total_versions}")
    return version_spreads

def calculate_criteria(input_file, output_file):
    """Main criteria calculation pipeline"""
    
    print("="*60)
    print("STEP 05: CRITERIA CALCULATION")
    print("="*60)
    
    # Load topics
    print(f"Loading topic assignments from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} reviews")
    
    # Exclude outliers
    df = df[df['topic'] != -1]
    print(f"Excluding outliers, {len(df):,} reviews remaining")
    
    num_topics = df['topic'].nunique()
    print(f"Processing {num_topics} topics")
    
    # Calculate all criteria
    freq = calculate_topic_frequency(df)
    importance = calculate_user_importance(df)
    recency = calculate_review_recency(df)
    spread = calculate_version_spread(df)
    
    # Combine into dataframe
    print("Combining criteria into dataframe...")
    
    criteria_df = pd.DataFrame({
        'topic_id': list(freq.keys()),
        'topic_frequency': list(freq.values()),
        'user_importance_strength': [importance[tid] for tid in freq.keys()],
        'review_recency': [recency[tid] for tid in freq.keys()],
        'app_version_spread': [spread[tid] for tid in freq.keys()]
    })
    
    # Sort by topic_id
    criteria_df = criteria_df.sort_values('topic_id').reset_index(drop=True)
    
    # Statistics
    print("\n" + "="*60)
    print("CRITERIA CALCULATION SUMMARY")
    print("="*60)
    print(f"Topics processed: {len(criteria_df)}")
    print(f"\nCriteria statistics:")
    print(criteria_df[['topic_frequency', 'user_importance_strength', 
                       'review_recency', 'app_version_spread']].describe())
    
    print(f"\nTop 5 topics by frequency:")
    top_freq = criteria_df.nlargest(5, 'topic_frequency')
    for _, row in top_freq.iterrows():
        print(f"  Topic {row['topic_id']:3.0f}: {row['topic_frequency']:.4f}")
    
    print(f"\nTop 5 topics by user importance:")
    top_imp = criteria_df.nlargest(5, 'user_importance_strength')
    for _, row in top_imp.iterrows():
        print(f"  Topic {row['topic_id']:3.0f}: {row['user_importance_strength']:.4f}")
    
    # Save results
    print(f"Saving criteria scores to: {output_file}")
    criteria_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved criteria for {len(criteria_df)} topics")
    
    return True

if __name__ == "__main__":
    success = calculate_criteria(config.TOPICS_FILE, config.CRITERIA_FILE)
    
    if success:
        print("\n" + "="*60)
        print("CRITERIA CALCULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 06_ahp_weighting.py")
    else:
        print("\n" + "="*60)
        print("CRITERIA CALCULATION FAILED")
        print("="*60)
        sys.exit(1)
