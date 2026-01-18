"""
MSA Normalization Validation - Direct Sampling from DataFrame
Use this if you already have your reviews loaded in memory
"""

import pandas as pd
import numpy as np

def sample_for_validation(reviews_df, n_samples=385, random_seed=42):
    """
    Generate validation sample from existing reviews dataframe
    
    Parameters:
    -----------
    reviews_df : pandas.DataFrame
        Your reviews with columns: App Name, Rating, cleaned_review, MSA, etc.
    n_samples : int
        Number of reviews to sample (default: 385)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    validation_df : pandas.DataFrame
        Stratified sample ready for rater evaluation
    """
    
    print(f"{'='*60}")
    print("GENERATING MSA VALIDATION SAMPLE")
    print(f"{'='*60}")
    
    # Check required columns
    required_cols = ['cleaned_review', 'MSA', 'Rating']
    missing = [col for col in required_cols if col not in reviews_df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None
    
    print(f"Original dataset: {len(reviews_df):,} reviews")
    
    # Remove invalid rows
    df = reviews_df.dropna(subset=['cleaned_review', 'MSA']).copy()
    df = df[(df['cleaned_review'].str.strip() != '') & (df['MSA'].str.strip() != '')]
    print(f"✓ Valid reviews: {len(df):,}")
    
    # Stratified sampling by rating
    print(f"Target sample: {n_samples} reviews")
    print(f"Original rating distribution:")
    rating_dist = df['Rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        pct = count / len(df) * 100
        print(f"  {rating}★: {count:,} ({pct:.1f}%)")
    
    # Sample proportionally
    samples = []
    for rating in sorted(df['Rating'].unique()):
        rating_df = df[df['Rating'] == rating]
        proportion = len(rating_df) / len(df)
        n_for_rating = int(n_samples * proportion)
        
        if n_for_rating == 0 and len(rating_df) > 0:
            n_for_rating = min(5, len(rating_df))
        
        if len(rating_df) <= n_for_rating:
            sample = rating_df
        else:
            sample = rating_df.sample(n=n_for_rating, random_state=random_seed)
        
        samples.append(sample)
    
    # Combine
    sample_df = pd.concat(samples, ignore_index=True)
    
    # Adjust to exact size
    if len(sample_df) < n_samples:
        remaining = df[~df.index.isin(sample_df.index)]
        additional = remaining.sample(n=n_samples - len(sample_df), random_state=random_seed)
        sample_df = pd.concat([sample_df, additional], ignore_index=True)
    elif len(sample_df) > n_samples:
        sample_df = sample_df.sample(n=n_samples, random_state=random_seed)
    
    # Shuffle
    sample_df = sample_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Sampled {len(sample_df)} reviews")
    print(f"\nSampled rating distribution:")
    sample_dist = sample_df['Rating'].value_counts().sort_index()
    for rating, count in sample_dist.items():
        pct = count / len(sample_df) * 100
        print(f"  {rating}★: {count} ({pct:.1f}%)")
    
    # Prepare validation columns
    validation_df = sample_df.copy()
    validation_df['review_id'] = range(1, len(validation_df) + 1)
    validation_df['rater1_acceptable'] = ''
    validation_df['rater2_acceptable'] = ''
    validation_df['notes'] = ''
    
    # Reorder and rename columns for clarity
    cols_map = {
        'cleaned_review': 'Original_Dialectal',
        'MSA': 'Normalized_MSA',
        'App Name': 'App_Name',
        'App Version': 'App_Version'
    }
    
    validation_df = validation_df.rename(columns=cols_map)
    
    # Select final columns
    final_cols = ['review_id', 'App_Name', 'Rating', 'Original_Dialectal', 
                  'Normalized_MSA', 'rater1_acceptable', 'rater2_acceptable', 'notes']
    
    # Add optional columns if they exist
    optional_cols = ['App_Version', 'Date', 'Source', 'MSA_Sentiment_Label']
    for col in optional_cols:
        if col in validation_df.columns:
            final_cols.insert(-3, col)  # Insert before rater columns
    
    available_cols = [col for col in final_cols if col in validation_df.columns]
    validation_df = validation_df[available_cols]
    
    # Statistics
    print(f"Sample statistics:")
    print(f"  Mean review length: {validation_df['Original_Dialectal'].str.split().str.len().mean():.1f} words")
    print(f"  Apps in sample: {validation_df['App_Name'].nunique()}")
    
    return validation_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # If you have reviews already loaded, use directly:
    # reviews = pd.read_csv("your_reviews.csv")  # Or use your existing dataframe
    validation_sample = sample_for_validation(reviews)
    validation_sample.to_csv("msa_validation_sample.csv", index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("READY TO USE")
    print("="*60)
    print("\nTo generate sample:")
    print("  1. Load your reviews dataframe")
    print("  2. Call: sample = sample_for_validation(reviews_df)")
    print("  3. Save: sample.to_csv('validation.csv', index=False, encoding='utf-8-sig')")
    print("\nOr in one line:")
    print("  sample_for_validation(reviews).to_csv('validation.csv', index=False, encoding='utf-8-sig')")