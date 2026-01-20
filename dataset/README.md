## Required File

Place your reviews CSV file here as: **`reviews.csv`**

## Expected Format

The CSV file must contain the following columns:

| Column Name     | Type   | Description                                      | Example                          |
|----------------|--------|--------------------------------------------------|----------------------------------|
| `review_text`  | string | The review content in Arabic                     | "التطبيق ممتاز وسهل الاستخدام"    |
| `rating`       | int    | Star rating (1-5)                                | 5                                |
| `date`         | string | Review submission date (YYYY-MM-DD)              | 2024-08-01                       |
| `app_version`  | string | App version at time of review                    | 3.2.1                            |
| `app_name`     | string | Name of the FemTech app                          | Flo Period Tracker               |

## Optional Columns

Additional columns that may be present but are not required:
- `review_id`: Unique identifier for the review
- `user_id`: Anonymous user identifier
- `helpful_count`: Number of users who found the review helpful
- `platform`: Android or iOS
- `country`: Country storefront

## Data Collection Notes

### Source
- Reviews were collected from Google Play Store and Apple App Store
- Collection date: August 01, 2025
- Geographic scope: 16 Arabic-speaking MENA countries

### Filtering Applied
- Only reviews written in Arabic were included
- Apps must be related to women's reproductive health (FemTech)
- Reviews must be publicly accessible

### Privacy and Ethics
- All reviews are publicly available and anonymized
- No personally identifiable information (PII) is included
- User IDs (if present) are anonymized by app stores
- This dataset is for academic research purposes only

## Sample Data Structure

```csv
review_text,rating,date,app_version,app_name
"التطبيق رائع ومفيد جداً",5,2024-07-15,3.2.1,Flo
"مشكلة في تسجيل الدخول",1,2024-07-20,3.2.0,Clue
"محتوى مفيد للحمل",4,2024-07-25,2.1.5,Ovia
```

## Data Statistics (Expected)

After preprocessing, the dataset should contain:
- **Total reviews**: ~27,000
- **Date range**: 2011-2025
- **Apps**: ~51 FemTech apps
- **Platforms**: Google Play (96%), App Store (4%)
- **Languages**: Arabic (all dialects normalized to MSA)

## Usage

The raw `reviews.csv` file will be processed by:
1. `01_data_preprocessing.py` - Clean and filter
2. `02_dialect_normalization.py` - Convert to MSA
3. Subsequent analysis scripts

## Data Not Included

Due to privacy and copyright considerations, the actual review dataset is **NOT** included in this repository. Researchers wishing to replicate this study should:

1. Collect their own reviews from public app stores
2. Ensure compliance with app store terms of service
3. Respect user privacy and anonymize any identifiable information

## Contact

For questions about data collection methodology, or the original dataset used in our study please contact me at (with a reasonable request):
**Waad Alhoshan** (wmaboud@imamu.edu.sa)
