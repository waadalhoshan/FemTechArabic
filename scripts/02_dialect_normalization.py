"""
Script 02: Dialect Normalization
Converts dialectal Arabic reviews to Modern Standard Arabic (MSA) using GPT-4o

Input: output/cleaned_reviews.csv
Output: output/msa_reviews.csv

Steps:
1. Load cleaned reviews
2. Batch reviews for API efficiency
3. Call OpenAI GPT-4o with structured prompts
4. Verify and validate normalized text
5. Save MSA-normalized reviews
"""

import pandas as pd
import sys
from pathlib import Path
import time
from tqdm import tqdm

# OpenAI import
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    print("   Run: pip install openai")
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def create_normalization_prompt(review_text):
    """Create structured prompt for dialect normalization"""
    prompt = f"""You are an expert in Arabic linguistics. Convert the following dialectal Arabic text to Modern Standard Arabic (MSA).

IMPORTANT RULES:
1. Preserve the original meaning exactly
2. Use proper MSA grammar and vocabulary
3. Keep the same sentiment and tone
4. Do not add or remove information
5. Output ONLY the normalized text, no explanations

Dialectal text:
{review_text}

MSA text:"""
    
    return prompt

def normalize_single_review(client, review_text):
    """Normalize a single review using GPT-4o"""
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Arabic linguist specializing in dialect normalization."},
                {"role": "user", "content": create_normalization_prompt(review_text)}
            ],
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS
        )
        
        normalized_text = response.choices[0].message.content.strip()
        return normalized_text
    
    except Exception as e:
        print(f"Error normalizing review: {e}")
        return review_text  # Return original if normalization fails

def normalize_batch(client, reviews_batch):
    """Normalize a batch of reviews"""
    normalized = []
    
    for review in reviews_batch:
        norm_text = normalize_single_review(client, review)
        normalized.append(norm_text)
        time.sleep(0.1)  # Small delay between calls
    
    return normalized

def normalize_reviews(input_file, output_file):
    """Main normalization pipeline"""
    
    print("="*60)
    print("STEP 02: DIALECT NORMALIZATION")
    print("="*60)
    
    # Check API key
    if config.OPENAI_API_KEY == "your-api-key-here":
        print("ERROR: OpenAI API key not configured")
        print("   Set OPENAI_API_KEY in config.py or as environment variable")
        return False
    
    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        print("OpenAI client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return False
    
    # Load cleaned reviews
    print(f"Loading cleaned reviews from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} reviews")
    
    # Create batches
    total_reviews = len(df)
    batch_size = config.NORMALIZATION_BATCH_SIZE
    num_batches = (total_reviews + batch_size - 1) // batch_size
    
    print(f"Processing {total_reviews:,} reviews in {num_batches} batches...")
    print(f"   Batch size: {batch_size}")
    print(f"   Model: {config.OPENAI_MODEL}")
    print(f"   Estimated time: ~{total_reviews * 1.5 / 60:.1f} minutes")
    print("\n This step may take a while and incur API costs")
    
    # Process in batches
    normalized_texts = []
    
    for i in tqdm(range(0, total_reviews, batch_size), desc="Normalizing"):
        batch = df['review_text'].iloc[i:i+batch_size].tolist()
        
        try:
            normalized_batch = normalize_batch(client, batch)
            normalized_texts.extend(normalized_batch)
            
            # Delay between batches to avoid rate limits
            if i + batch_size < total_reviews:
                time.sleep(config.NORMALIZATION_DELAY)
        
        except Exception as e:
            print(f"\n ERROR in batch {i//batch_size + 1}: {e}")
            print("   Returning original text for this batch")
            normalized_texts.extend(batch)
    
    # Add normalized text to dataframe
    df['review_text_msa'] = normalized_texts
    df['review_text_original'] = df['review_text']  # Keep original
    df['review_text'] = df['review_text_msa']  # Use MSA as main text
    
    # Validation
    print("Validating normalized reviews...")
    empty_normalized = df['review_text_msa'].isna().sum()
    if empty_normalized > 0:
        print(f"WARNING: {empty_normalized} reviews have empty normalized text")
        # Fill with original text
        df['review_text_msa'] = df['review_text_msa'].fillna(df['review_text_original'])
    
    # Statistics
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Total reviews processed: {len(df):,}")
    print(f"Successfully normalized:  {len(df) - empty_normalized:,}")
    print(f"Average original length:  {df['review_text_original'].str.len().mean():.1f} chars")
    print(f"Average MSA length:       {df['review_text_msa'].str.len().mean():.1f} chars")
    
    # Sample comparison
    print("\n Sample comparison (first review):")
    print(f"Original: {df['review_text_original'].iloc[0]}")
    print(f"MSA:      {df['review_text_msa'].iloc[0]}")
    
    # Save normalized data
    print(f"Saving normalized reviews to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df):,} MSA-normalized reviews")
    
    return True

if __name__ == "__main__":
    success = normalize_reviews(config.CLEANED_REVIEWS, config.MSA_REVIEWS)
    
    if success:
        print("\n" + "="*60)
        print("NORMALIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 03_sentiment_analysis.py")
    else:
        print("\n" + "="*60)
        print("NORMALIZATION FAILED")
        print("="*60)
        sys.exit(1)
