"""
Script 04: Topic Modeling
Extracts topics from MSA reviews using BERTopic with AraBERT embeddings

Input: output/sentiment_results.csv
Output: output/topics.csv

Steps:
1. Load reviews with sentiment labels
2. Generate embeddings using AraBERTv2
3. Apply BERTopic with HDBSCAN clustering
4. Perform semantic topic reduction
5. Calculate coherence scores
6. Export topics with keywords and assignments
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# BERTopic imports
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    print("ERROR: Required packages not installed")
    print("   Run: pip install bertopic sentence-transformers umap-learn hdbscan")
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

def initialize_embedding_model():
    """Initialize AraBERT model for embeddings"""
    print(f"Loading embedding model: {config.ARABERT_MODEL}")
    try:
        model = SentenceTransformer(config.ARABERT_MODEL)
        print("Embedding model loaded")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return None

def create_bertopic_model(embedding_model):
    """Create BERTopic model with custom components"""
    print("Configuring BERTopic model...")
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=config.RANDOM_STATE
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.MIN_TOPIC_SIZE,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # CountVectorizer for keyword extraction
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None  # No stop words for Arabic
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=None,  # Will reduce later
        calculate_probabilities=True,
        verbose=True
    )
    
    print("BERTopic model configured")
    return topic_model

def calculate_coherence(topic_model, docs):
    """Calculate topic coherence score (c_v)"""
    # Note: Full coherence calculation requires gensim
    # This is a simplified version
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora import Dictionary
        
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in docs]
        dictionary = Dictionary(tokenized_docs)
        
        # Get topics
        topics = topic_model.get_topics()
        topic_words = []
        for topic_id in topics:
            if topic_id != -1:  # Skip outlier topic
                words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
                topic_words.append(words)
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        return coherence_score
    
    except Exception as e:
        print(f"Could not calculate coherence: {e}")
        return 0.0

def extract_topics(input_file, output_file):
    """Main topic modeling pipeline"""
    
    print("="*60)
    print("STEP 04: TOPIC MODELING")
    print("="*60)
    
    # Load reviews
    print(f"Loading reviews from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} reviews")
    
    # Prepare documents
    docs = df['review_text'].tolist()
    
    # Initialize embedding model
    embedding_model = initialize_embedding_model()
    if embedding_model is None:
        return False
    
    # Generate embeddings
    print(f"Generating embeddings for {len(docs):,} documents...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    print(f"Generated embeddings of shape {embeddings.shape}")
    
    # Create and fit BERTopic model
    topic_model = create_bertopic_model(embedding_model)
    
    print(f"Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    initial_topics = len(set(topics)) - 1  # Exclude -1 (outliers)
    print(f"Initial topics extracted: {initial_topics}")
    
    # Topic reduction
    print(f"Performing semantic topic reduction to {config.NR_TOPICS} topics...")
    topic_model.reduce_topics(docs, nr_topics=config.NR_TOPICS)
    topics = topic_model.topics_
    
    final_topics = len(set(topics)) - 1
    print(f"Final topics after reduction: {final_topics}")
    
    # Calculate coherence
    print("Calculating topic coherence...")
    coherence = calculate_coherence(topic_model, docs)
    print(f"✓ Coherence score (c_v): {coherence:.4f}")
    
    # Get topic information
    print("Extracting topic information...")
    topic_info = topic_model.get_topic_info()
    
    # Add topics and probabilities to dataframe
    df['topic'] = topics
    df['topic_probability'] = [max(prob) if isinstance(prob, np.ndarray) else prob for prob in probs]
    
    # Topic statistics
    print("\n" + "="*60)
    print("TOPIC MODELING SUMMARY")
    print("="*60)
    print(f"Total documents:      {len(docs):,}")
    print(f"Initial topics:       {initial_topics}")
    print(f"Final topics:         {final_topics}")
    print(f"Coherence (c_v):      {coherence:.4f}")
    print(f"\nTop 5 largest topics:")
    topic_counts = df['topic'].value_counts().head(5)
    for topic_id, count in topic_counts.items():
        if topic_id != -1:
            keywords = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
            print(f"  Topic {topic_id:3d}: {count:5d} reviews - {', '.join(keywords)}")
    
    outliers = (df['topic'] == -1).sum()
    print(f"\nOutliers (topic -1):  {outliers:,} ({100*outliers/len(df):.1f}%)")
    
    # Save results
    print(f"Saving topic assignments to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {len(df):,} reviews with topic assignments")
    
    # Save topic model
    model_path = output_file.replace('.csv', '_model')
    print(f"Saving BERTopic model to: {model_path}")
    topic_model.save(model_path)
    print("Model saved")
    
    # Save topic info
    topic_info_file = output_file.replace('.csv', '_info.csv')
    topic_info.to_csv(topic_info_file, index=False, encoding='utf-8-sig')
    print(f"Topic info saved to: {topic_info_file}")
    
    return True

if __name__ == "__main__":
    success = extract_topics(config.SENTIMENT_RESULTS, config.TOPICS_FILE)
    
    if success:
        print("\n" + "="*60)
        print("✅ TOPIC MODELING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext step: Run 05_criteria_calculation.py")
    else:
        print("\n" + "="*60)
        print("❌ TOPIC MODELING FAILED")
        print("="*60)
        sys.exit(1)
