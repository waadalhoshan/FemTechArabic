"""
Script 01: Data Preprocessing
Cleans and filters raw Arabic reviews from FemTech apps

Input: dataset/reviews.csv
Output: output/cleaned_reviews.csv

Steps:
1. Load raw reviews
2. Remove empty or very short reviews (<3 words)
3. Remove emoji-only reviews
4. Remove excessive punctuation/symbols
5. Basic text normalization
6. Save cleaned dataset
"""

import pandas as pd
import re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def remove_excessive_punctuation(text):
    """Remove sequences of punctuation/symbols without actual words"""
    # Check if text has at least some Arabic letters
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    if not arabic_pattern.search(text):
        return None
    
    # Remove excessive repeated punctuation (more than 3 in a row)
    text = re.compile(r'([!?.,،؛]){4,}').sub(r'\1\1\1', text)
    
    return text

def count_words(text):
    """Count Arabic words in text"""
    if pd.isna(text):
        return 0
    # Arabic word pattern
    words = re.findall(r'[\u0600-\u06FF]+', str(text))
    return len(words)

def is_emoji_only(text):
    """Check if review contains only emojis/symbols without text"""
    if pd.isna(text):
        return True
    
    # Remove all emojis and symbols
    text_clean = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
    text_clean = text_clean.strip()
    
    return len(text_clean) == 0

def preprocess_reviews(input_file, output_file):
    """Main preprocessing pipeline"""
    
    print("="*60)
    print("STEP 01: DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} reviews")
    
    initial_count = len(df)
    
    # Check required columns
    required_cols = ['review_text', 'rating', 'date', 'app_version', 'app_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    # Step 1: Remove null reviews
    print("Step 1: Removing null/empty reviews...")
    df = df.dropna(subset=['review_text'])
    df = df[df['review_text'].str.strip() != '']
    print(f"   Removed {initial_count - len(df):,} null reviews")
    print(f"   Remaining: {len(df):,} reviews")
    
    # Step 2: Remove emoji-only reviews
    if config.REMOVE_EMOJI_ONLY:
        print("Step 2: Removing emoji-only reviews...")
        before = len(df)
        df['is_emoji_only'] = df['review_text'].apply(is_emoji_only)
        df = df[~df['is_emoji_only']]
        df = df.drop('is_emoji_only', axis=1)
        print(f"   Removed {before - len(df):,} emoji-only reviews")
        print(f"   Remaining: {len(df):,} reviews")
    
    # Step 3: Clean excessive punctuation
    if config.REMOVE_EXCESSIVE_PUNCT:
        print("Step 3: Cleaning excessive punctuation...")
        df['review_text'] = df['review_text'].apply(remove_excessive_punctuation)
        df = df.dropna(subset=['review_text'])
        print(f"   Remaining: {len(df):,} reviews")
    
    # Step 4: Count words and filter short reviews
    print(f"Step 4: Filtering reviews with < {config.MIN_REVIEW_LENGTH} words...")
    df['word_count'] = df['review_text'].apply(count_words)
    before = len(df)
    df = df[df['word_count'] >= config.MIN_REVIEW_LENGTH]
    print(f"   Removed {before - len(df):,} short reviews")
    print(f"   Remaining: {len(df):,} reviews")
    
    # Step 5: Basic normalization
    print("Step 5: Basic text normalization...")
    # Remove extra whitespace
    df['review_text'] = df['review_text'].str.strip()
    df['review_text'] = df['review_text'].str.replace(r'\s+', ' ', regex=True)
    
    # Normalize Arabic characters (optional)
    df['review_text'] = df['review_text'].str.replace('ي', 'ى')  # Normalize Alef Maksura
    df['review_text'] = df['review_text'].str.replace('ة', 'ه')  # Normalize Ta Marbuta
    
    # Step 6: Convert date to datetime
    print("Step 6: Converting date format...")
    df['date'] = pd.to_datetime(df['date'], format=config.DATE_FORMAT, errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"   Remaining: {len(df):,} reviews with valid dates")
    
    # Step 7: Validate ratings
    print("Step 7: Validating ratings...")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'].between(1, 5)]
    print(f"   Remaining: {len(df):,} reviews with valid ratings (1-5)")
    
    # Final statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Initial reviews:    {initial_count:,}")
    print(f"Final reviews:      {len(df):,}")
    print(f"Removed:            {initial_count - len(df):,} ({100*(initial_count-len(df))/initial_count:.1f}%)")
    print(f"\nRating distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Apps: {df['app_name'].nunique()}")
    
    # Save cleaned data
    print(f"Saving cleaned data to: {output_file}")
    config.create_directories()
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df):,} cleaned reviews")
    
    return True

if __name__ == "__main__":
    success = preprocess_reviews(config.INPUT_REVIEWS, config.CLEANED_REVIEWS)
    
    if success:
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 02_dialect_normalization.py")
    else:
        print("\n" + "="*60)
        print("PREPROCESSING FAILED")
        print("="*60)
        sys.exit(1)
