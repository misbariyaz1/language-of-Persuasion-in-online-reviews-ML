# The Language of Persuasion in Online Reviews
## Predicting Review Helpfulness Using Linguistic Features and NLP

## Overview
This repository contains code for Problem Set 2 of the SSIM916 module (Machine Learning for Social Data Science). The project investigates what linguistic features make online food reviews more persuasive, using the Amazon Fine Food Reviews dataset. Four machine learning models are built and compared — Logistic Regression (linguistic features only), Logistic Regression (TF-IDF + linguistic), Random Forest, and Gradient Boosting — to predict whether a review is perceived as helpful based on its language, structure, and sentiment.

## Dataset
The dataset used is the **Amazon Fine Food Reviews** dataset, available publicly on Kaggle:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

File: `Reviews.csv`

Key variables:
- `Text`: raw review body (primary input for NLP)
- `HelpfulnessNumerator`: number of users who found the review helpful
- `HelpfulnessDenominator`: total number of users who voted on the review
- `Score`: star rating (1–5)
- `Summary`: short review title
- `helpful`: derived binary label (1 = helpful if ratio ≥ 0.6, else 0)

## Research Question
What linguistic features make online reviews more persuasive, and can these features predict perceived helpfulness?

## Project Structure
```
├── README.md
├── Reviews.csv                  # dataset (download from Kaggle — not included in repo)
└── analysis_code.ipynb                  # full pipeline: cleaning → EDA → features → models
```

## How to Reproduce

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud scipy
```

### 2. Download NLTK corpora
The script handles this automatically on first run, but you can also run manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
```

### 3. Download the dataset
- Go to https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Download `Reviews.csv`
- Place it in the **same folder** as `analysis_code.ipynb

### 4. Run the script
```bash
python analysis_code.ipynb
```

This will:
- Load and clean the data (deduplication, HTML removal, filtering)
- Create the binary helpfulness label
- Run EDA and save all figures to the working directory
- Engineer 18 linguistic features
- Train and evaluate all 4 models
- Print a final model comparison table

### 5. Expected output
All  figures are saved as `.png` files in the working directory,. Runtime is approximately 10–15 minutes on a standard laptop depending on hardware.

## Key Features Engineered
| Group | Features |
|---|---|
| Structural | char_length, word_count, sentence_count, avg_word_length, avg_sentence_length |
| Rhetorical | exclamation_ratio, question_ratio, first_person_ratio, uppercase_ratio |
| Specificity | contains_comparison, specificity_score |
| Credibility | type_token_ratio, hedging_count |
| Sentiment (VADER) | sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu |
| Other | star_rating |


## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk
- wordcloud
- scipy

Merge the 4 review data into once before using or else you can downlaod from kaggle link is proivded below 
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews