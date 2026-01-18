

# Detailed Methodology

This document provides additional methodological details for the FemTech Arabic reviews analysis pipeline.

## Overview

The analysis follows a systematic 7-step process to identify, extract, and prioritize user concerns and perspectives from Arabic-language app reviews.

---

## Step 1: Data Preprocessing

### Objective
Clean raw reviews to ensure data quality for downstream analysis.

### Process
1. **Null Removal**: Remove empty or missing review texts
2. **Length Filtering**: Exclude reviews with fewer than 3 Arabic words
3. **Emoji Filtering**: Remove emoji-only reviews without textual content
4. **Punctuation Cleaning**: Remove excessive repeated punctuation
5. **Date Validation**: Convert and validate review timestamps
6. **Rating Validation**: Ensure ratings are within 1-5 range

### Rationale
Short, emoji-only, or symbol-heavy reviews lack semantic content for topic modeling and may introduce noise.

### Expected Output
~96% of raw reviews retained after filtering (based on paper results).

---

## Step 2: Dialect Normalization

### Objective
Convert diverse Arabic dialects to Modern Standard Arabic (MSA) for linguistic consistency.

### Process
Uses OpenAI GPT-4o with structured prompts:
- Preserves original meaning and sentiment
- Standardizes grammar and vocabulary
- Maintains cultural context
- Validates output for completeness

### Prompt Structure
```
You are an expert in Arabic linguistics. Convert the following dialectal 
Arabic text to Modern Standard Arabic (MSA).

IMPORTANT RULES:
1. Preserve the original meaning exactly
2. Use proper MSA grammar and vocabulary  
3. Keep the same sentiment and tone
4. Do not add or remove information
5. Output ONLY the normalized text, no explanations

Dialectal text: [USER_REVIEW]
MSA text:
```

### Rationale
Dialectal variation fragments semantically similar feedback across topics. Normalization improves topic coherence by 10-15% (AlShargi, 2020; Bourahouat et al., 2025).

### API Configuration
- Model: GPT-4o
- Temperature: 0.3 (low creativity, high consistency)
- Max tokens: 150
- Batch processing with rate limit handling

---

## Step 3: Sentiment Analysis

### Objective
Classify emotional tone of reviews (positive/neutral/negative) with confidence scores.

### Model
**CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment**
- Fine-tuned Arabic BERT for sentiment classification
- Trained on dialectal and MSA user-generated content
- Returns label + confidence score per review

### Process
1. Load pre-trained transformer model
2. Batch process reviews (GPU-accelerated if available)
3. Extract:
   - Sentiment label (positive/neutral/negative)
   - Confidence score (0-1)
   - Individual class probabilities

### Validation
Sentiment distribution aligns with star ratings:
- 1-star reviews: predominantly negative
- 5-star reviews: predominantly positive
- High model confidence at extreme ratings

---

## Step 4: Topic Modeling

### Objective
Extract coherent themes from review corpus using semantic clustering.

### Method: BERTopic
**Components**:
1. **Embeddings**: AraBERTv2 (MSA-trained transformer)
2. **Dimensionality Reduction**: UMAP (n_components=5)
3. **Clustering**: HDBSCAN (min_cluster_size=10)
4. **Keyword Extraction**: CountVectorizer (unigrams + bigrams)

### Process
1. Generate semantic embeddings for all reviews
2. Apply UMAP to reduce dimensionality (512→5 dimensions)
3. Cluster similar reviews using HDBSCAN
4. Extract representative keywords per topic
5. Reduce topics via semantic merging (353→50 topics)
6. Calculate coherence score (c_v) to validate quality

### Topic Reduction Strategy
- Initial model produces 350+ fine-grained topics
- Semantic reduction iteratively merges similar topics
- Optimal configuration: K=50 (c_v = 0.4135)
- Balance between specificity and interpretability

### Manual Coding
After automated extraction:
- 3 experts review topics independently
- Assign descriptive labels based on keywords + sample reviews
- Classify as "concern" or "perspective"
- Inter-rater reliability: κ = 0.75 (substantial agreement)

---

## Step 5: Criteria Calculation

### Objective
Quantify four dimensions of topic importance.

### Criteria Formulas

#### 1. Topic Frequency
```
Frequency_i = n_i / N
```
- n_i = number of reviews mentioning topic i
- N = total reviews
- **Interpretation**: Prevalence in user feedback

#### 2. User Importance Strength
```
Importance_i = mean(|rating - 3| × sentiment × confidence)
```
- rating ∈ {1,2,3,4,5}
- sentiment ∈ {-1, 0, +1}
- confidence ∈ [0, 1]
- **Interpretation**: Intensity of user opinion (dissatisfaction or satisfaction)

#### 3. Review Recency
```
Recency_i = mean(1 - (days_since_review / max_age))
```
- Normalized to [0, 1]
- Recent reviews score closer to 1
- **Interpretation**: Current relevance

#### 4. App Version Spread
```
Spread_i = unique_versions_i / total_versions
```
- Measures persistence across app updates
- **Interpretation**: Long-standing vs. transient issues

### Rationale
Multi-criteria approach captures:
- **Frequency**: What many users mention
- **Importance**: What users care deeply about
- **Recency**: What matters now
- **Spread**: What persists over time

---

## Step 6: AHP Weighting

### Objective
Determine relative importance of evaluation criteria through expert judgment.

### Method: Analytic Hierarchy Process (AHP)

#### Process
1. **Pairwise Comparisons**: 3 experts compare criteria using Saaty's 1-9 scale
2. **Priority Vectors**: Calculate using eigenvalue method
3. **Consistency Check**: CR < 0.10 considered acceptable
4. **Aggregation**: Geometric mean of expert weights

#### Saaty's Scale
| Score | Meaning |
|-------|---------|
| 1 | Equal importance |
| 3 | Moderate importance |
| 5 | Strong importance |
| 7 | Very strong importance |
| 9 | Extreme importance |

#### Final Weights (from paper)
- **Topic Frequency**: 0.3803
- **User Importance Strength**: 0.3441
- **Review Recency**: 0.1117
- **App Version Spread**: 0.1639

#### Consistency Ratios
- Expert 1: CR = 0.0170 ✓
- Expert 2: CR = 0.0076 ✓
- Expert 3: CR = 0.0780 ✓
- Group: CR = 0.0147 ✓

All CR values < 0.10 indicate logically consistent judgments.

---

## Step 7: TOPSIS Ranking

### Objective
Rank topics by similarity to ideal solution across weighted criteria.

### Method: TOPSIS
(Technique for Order Preference by Similarity to Ideal Solution)

#### Process
1. **Normalize**: Vector normalization of decision matrix
2. **Weight**: Multiply by AHP weights
3. **Ideal Solutions**:
   - A+ (ideal): Best value per criterion
   - A− (negative-ideal): Worst value per criterion
4. **Separation**: Euclidean distance from A+ and A−
5. **Closeness**: C_i = S_i− / (S_i+ + S_i−)
6. **Rank**: Sort by closeness coefficient (higher = better)

#### Mathematical Formulation
```
1. Normalize:     r_ij = x_ij / √(Σx_ij²)
2. Weight:        v_ij = w_j × r_ij
3. Ideal:         A+ = {max(v_ij)} for benefit criteria
                  A− = {min(v_ij)} for benefit criteria
4. Separation:    S_i+ = √Σ(v_ij - A+_j)²
                  S_i− = √Σ(v_ij - A−_j)²
5. Closeness:     C_i = S_i− / (S_i+ + S_i−)
```

#### Interpretation
- TOPSIS score ∈ [0, 1]
- Higher scores = closer to ideal, farther from worst
- Rank 1 = highest priority theme

---

## Key Findings (from paper)

### Top-Ranked Themes
1. **Widespread user approval and gratitude** (TOPSIS: 0.8719)
   - Positive perspective
   - High frequency + importance + recency
2. **Easy-to-use tracking and interface** (TOPSIS: 0.5817)
   - Positive perspective  
   - Strong usability satisfaction
3. **Poor Arabic language support** (TOPSIS: 0.5720)
   - User concern
   - Persistent across versions

### Insights
- Positive perspectives dominate top ranks
- Language support is critical concern
- Monetization barriers (ads/paywalls) affect access
- Prediction accuracy concerns affect fewer users but matter deeply

---

## Validation & Reliability

### Inter-Rater Agreement
- **Topic Labels** (Likert scale): Weighted κ = 0.796 (substantial)
- **Concern/Perspective**: κ = 0.75, κ = 0.71 (substantial)
- **Significance**: p < .0001

### Topic Coherence
- Initial model: c_v = 0.35
- Final model (K=50): c_v = 0.4135
- Improvement: +18% semantic consistency

### Sentiment-Rating Alignment
- 1-star reviews: 49.3% negative
- 5-star reviews: 85.1% positive
- Strong correlation validates sentiment model

---

## References

AlShargi, F. (2020). Arabic Dialect Normalization for Improving NLP Applications. University of Colorado Boulder.

Bourahouat, G., Abourezq, M., & Daoudi, N. (2025). Enhancing Arabic Topic Modeling Using BERTopic. In Communication and Information Technologies through the Lens of Innovation (pp. 129-137). Springer.

Saaty, T. L. (1980). The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation. McGraw-Hill.

---

## Contact

For methodological questions:
**Waad Alhoshan** (wmaboud@imamu.edu.sa)
