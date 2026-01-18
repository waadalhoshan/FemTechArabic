"""
Script 07: TOPSIS Ranking
Ranks topics using TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

Input: output/criteria_scores.csv, config.py (AHP_WEIGHTS)
Output: output/final_rankings.csv

Steps:
1. Load criteria scores
2. Normalize decision matrix
3. Apply AHP weights
4. Calculate ideal and negative-ideal solutions
5. Calculate separation measures
6. Calculate closeness coefficient
7. Rank topics
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def normalize_matrix(data):
    """
    Normalize decision matrix using vector normalization
    x_ij_norm = x_ij / sqrt(sum(x_ij^2))
    """
    data_squared = data ** 2
    sum_squared = data_squared.sum(axis=0)
    normalized = data / np.sqrt(sum_squared)
    return normalized

def calculate_weighted_matrix(normalized_data, weights):
    """
    Apply weights to normalized matrix
    v_ij = w_j * x_ij_norm
    """
    return normalized_data * weights

def calculate_ideal_solutions(weighted_data, criteria_types):
    """
    Calculate ideal (A+) and negative-ideal (A-) solutions
    For benefit criteria: A+ = max, A- = min
    For cost criteria: A+ = min, A- = max
    """
    ideal_positive = []
    ideal_negative = []
    
    for col, crit_type in zip(weighted_data.columns, criteria_types):
        if crit_type == 'benefit':
            ideal_positive.append(weighted_data[col].max())
            ideal_negative.append(weighted_data[col].min())
        else:  # cost
            ideal_positive.append(weighted_data[col].min())
            ideal_negative.append(weighted_data[col].max())
    
    return np.array(ideal_positive), np.array(ideal_negative)

def calculate_separation(weighted_data, ideal_solution):
    """
    Calculate Euclidean distance from ideal solution
    S_i = sqrt(sum((v_ij - v_j*)^2))
    """
    diff = weighted_data - ideal_solution
    squared_diff = diff ** 2
    separation = np.sqrt(squared_diff.sum(axis=1))
    return separation

def calculate_closeness(separation_positive, separation_negative):
    """
    Calculate closeness coefficient
    C_i = S_i- / (S_i+ + S_i-)
    """
    closeness = separation_negative / (separation_positive + separation_negative)
    return closeness

def rank_topics_topsis(input_file, output_file):
    """Main TOPSIS ranking pipeline"""
    
    print("="*60)
    print("STEP 07: TOPSIS RANKING")
    print("="*60)
    
    # Load criteria scores
    print(f"Loading criteria scores from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded criteria for {len(df)} topics")
    
    # Define criteria columns and weights
    criteria_cols = [
        'topic_frequency',
        'user_importance_strength',
        'review_recency',
        'app_version_spread'
    ]
    
    # Get weights from config
    weights = np.array([
        config.AHP_WEIGHTS['Topic_Frequency'],
        config.AHP_WEIGHTS['User_Importance_Strength'],
        config.AHP_WEIGHTS['Review_Recency'],
        config.AHP_WEIGHTS['App_Version_Spread']
    ])
    
    print(f"AHP Weights:")
    for col, weight in zip(criteria_cols, weights):
        print(f"  {col:30s}: {weight:.4f}")
    print(f"  Sum: {weights.sum():.4f}")
    
    # Extract decision matrix
    decision_matrix = df[criteria_cols].values
    
    # Step 1: Normalize matrix
    print(f"Step 1: Normalizing decision matrix...")
    normalized_matrix = normalize_matrix(decision_matrix)
    print(f"Matrix normalized")
    
    # Step 2: Apply weights
    print(f"Step 2: Applying AHP weights...")
    weighted_matrix = calculate_weighted_matrix(normalized_matrix, weights)
    weighted_df = pd.DataFrame(weighted_matrix, columns=criteria_cols)
    print(f"Weights applied")
    
    # Step 3: Calculate ideal solutions
    print(f"Step 3: Calculating ideal solutions...")
    criteria_types = [config.CRITERIA_TYPES[name] for name in 
                     ['Topic_Frequency', 'User_Importance_Strength', 
                      'Review_Recency', 'App_Version_Spread']]
    
    ideal_positive, ideal_negative = calculate_ideal_solutions(
        weighted_df, criteria_types
    )
    
    print(f"Ideal positive (A+): {ideal_positive}")
    print(f"Ideal negative (A-): {ideal_negative}")
    
    # Step 4: Calculate separation measures
    print(f"Step 4: Calculating separation measures...")
    separation_positive = calculate_separation(weighted_matrix, ideal_positive)
    separation_negative = calculate_separation(weighted_matrix, ideal_negative)
    print(f"Calculated S+ and S- for {len(df)} topics")
    
    # Step 5: Calculate closeness coefficient
    print(f"Step 5: Calculating closeness coefficients...")
    closeness = calculate_closeness(separation_positive, separation_negative)
    print(f"Calculated closeness coefficients")
    
    # Add results to dataframe
    df['separation_positive'] = separation_positive
    df['separation_negative'] = separation_negative
    df['topsis_score'] = closeness
    
    # Rank topics
    df['rank'] = df['topsis_score'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('rank')
    
    # Summary statistics
    print("\n" + "="*60)
    print("TOPSIS RANKING SUMMARY")
    print("="*60)
    print(f"Topics ranked: {len(df)}")
    print(f"TOPSIS score range: [{closeness.min():.4f}, {closeness.max():.4f}]")
    
    print(f"Top 10 Ranked Topics:")
    print(f"{'Rank':<6} {'Topic':<8} {'TOPSIS':<10} {'Frequency':<12} {'Importance':<12}")
    print("-" * 60)
    
    for _, row in df.head(10).iterrows():
        print(f"{row['rank']:<6.0f} "
              f"{row['topic_id']:<8.0f} "
              f"{row['topsis_score']:<10.4f} "
              f"{row['topic_frequency']:<12.4f} "
              f"{row['user_importance_strength']:<12.4f}")
    
    print(f"Bottom 5 Ranked Topics:")
    for _, row in df.tail(5).iterrows():
        print(f"{row['rank']:<6.0f} "
              f"{row['topic_id']:<8.0f} "
              f"{row['topsis_score']:<10.4f} "
              f"{row['topic_frequency']:<12.4f} "
              f"{row['user_importance_strength']:<12.4f}")
    
    # Correlation analysis
    print(f"Correlation between TOPSIS score and criteria:")
    for col in criteria_cols:
        corr = df['topsis_score'].corr(df[col])
        print(f"  {col:30s}: {corr:+.4f}")
    
    # Save results
    print(f"Saving final rankings to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved rankings for {len(df)} topics")
    
    # Save top topics separately
    top_file = output_file.replace('.csv', '_top10.csv')
    df.head(10).to_csv(top_file, index=False, encoding='utf-8-sig')
    print(f"Saved top 10 topics to: {top_file}")
    
    return True

if __name__ == "__main__":
    success = rank_topics_topsis(config.CRITERIA_FILE, config.FINAL_RANKINGS)
    
    if success:
        print("\n" + "="*60)
        print("TOPSIS RANKING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFULL ANALYSIS PIPELINE COMPLETED!")
        print(f"\nFinal results available in: {config.OUTPUT_DIR}/")
        print("  • final_rankings.csv - All ranked topics")
        print("  • final_rankings_top10.csv - Top 10 priorities")
    else:
        print("\n" + "="*60)
        print("TOPSIS RANKING FAILED")
        print("="*60)
        sys.exit(1)
