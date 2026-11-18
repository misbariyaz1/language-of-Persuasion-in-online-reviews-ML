# The Language of Persuasion in Online Reviews
## Predicting Review Helpfulness Using Linguistic Features and NLP

## Key Findings
What makes an online review helpful? I engineered 18 linguistic features and compared 4 machine learning models on 16,703 Amazon food reviews.

| Model | ROC-AUC | Accuracy | F1 (macro) |
|---|---|---|---|
| Logistic Regression: linguistic features only | 0.749 | 0.738 | 0.684 |
| **Logistic Regression: TF-IDF + linguistic (best)** | **0.761** | 0.737 | 0.683 |
| Random Forest: TF-IDF + linguistic | 0.728 | 0.706 | 0.648 |
| Gradient Boosting: TF-IDF + linguistic | 0.754 | 0.770 | 0.650 |

**What this means:** language predicts helpfulness only modestly. The best model reached 0.76 ROC-AUC, and adding TF-IDF text features improved on linguistic features alone by just 0.012. Accuracy is a weak measure here because 74% of reviews are labelled helpful, so ROC-AUC is the main metric.

## Overview
This project investigates what linguistic features make online food reviews more persuasive, using the Amazon Fine Food Reviews dataset. It was completed as part of my MSc studies for module **SSIM916 (Machine Learning for Social Data Science)**, a university module that teaches machine learning methods applied to social and behavioural data. Four models are built and compared to predict whether a review is perceived as helpful based on its language, structure and sentiment.

## Research Question
What linguistic features make online reviews more persuasive, and can these features predict perceived helpfulness?

## Dataset
**Amazon Fine Food Reviews** (568,454 reviews), publicly available on Kaggle:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

The original `Reviews.csv` is too large to upload to GitHub as one file, so it is split into 15 files (`review_data_1.csv` to `review_data_15.csv`). The script `combine_data.py` joins them back into `Reviews.csv`.

Key variables:
- `Text`: raw review body (primary input for NLP)
- `HelpfulnessNumerator`: number of users who found the review helpful
- `HelpfulnessDenominator`: total number of users who voted on the review
- `Score`: star rating (1 to 5)
- `Summary`: short review title
- `helpful`: derived binary label (1 = helpful if ratio is 0.6 or higher, else 0)

Cleaning steps: 568,454 raw reviews, 393,579 after removing duplicates, a random sample of 50,000, then 16,703 reviews that had at least 2 helpfulness votes.

## Project Structure
```
├── README.md
├── requirements.txt
├── combine_data.py              # joins the 15 data files into Reviews.csv
├── review_data_1.csv ... review_data_15.csv   # the dataset, split into chunks
└── analysis_code.ipynb          # full pipeline: cleaning, EDA, features, models
```

## How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Rebuild the dataset
```bash
python combine_data.py
```
This creates `Reviews.csv` in the same folder.

### 3. Download NLTK corpora
The notebook handles this on first run, or you can run it manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
```

### 4. Run the notebook
```bash
jupyter notebook analysis_code.ipynb
```

This will:
- Load and clean the data (deduplication, HTML removal, filtering)
- Create the binary helpfulness label
- Run EDA and save all figures to the working directory
- Engineer 18 linguistic features
- Train and evaluate all 4 models
- Print a final model comparison table

Runtime is approximately 10 to 15 minutes on a standard laptop.

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
Python 3 with the libraries listed in `requirements.txt`: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, wordcloud, scipy and jupyter.
