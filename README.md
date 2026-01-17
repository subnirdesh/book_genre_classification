# Book Genre Classification using Machine Learning

A comparative study of traditional machine learning and transformer-based approaches for automatic book genre classification using book descriptions from Goodreads.

## Project Overview

This project implements and compares two text classification approaches for predicting book genres:
- **Logistic Regression with TF-IDF**: A traditional machine learning baseline
- **DistilBERT**: A fine-tuned transformer model for contextual understanding

The dataset consists of 9,848 balanced book samples across 8 genres: Fiction, Mystery, Romance, Science Fiction, Fantasy, Horror, Biography, and Self-Help.

## Key Results

- **Logistic Regression**: 63.10% accuracy
- **DistilBERT**: 68.27% accuracy (5.17% improvement)
- DistilBERT showed significant improvements in Romance (+28%) and Biography (+20%)
- Both models performed best on vocabulary-distinct genres like Self-Help

## Dataset

The **Best Books Ever Dataset** was sourced from Goodreads and created by Lorena Casanova Lozano and Sergio Costa Planells (Universitat Oberta de Catalunya). The original data collection code is available on GitHub.

**Dataset Characteristics:**
- 9,848 books balanced across 8 genres
- Book descriptions averaging 888 characters
- Single-label classification after processing multi-genre entries

## Project Structure

```
book_genre_classification/
├── data/
│   ├── raw/                          # Original dataset (not tracked in Git)
│   └── processed/                    # Cleaned and balanced data (not tracked)
├── notebooks/
│   ├── 01_data_cleaning.ipynb       # Data preprocessing and balancing
│   ├── 02_train_logistic_regression.ipynb  # Traditional ML model
│   └── 03_train_Distillbert.ipynb   # Transformer model training
├── models/                           # Trained models (not tracked)
├── results/                          # Model predictions (not tracked)
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Conda (Anaconda or Miniconda)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/book_genre_classification.git
cd book_genre_classification
```

2. Create a conda environment:
```bash
conda create -n book_genre python=3.9
conda activate book_genre
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (required for preprocessing):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Dataset Setup

Due to file size limitations, the dataset is not included in this repository.

1. Download the Best Books Ever dataset from [source link]
2. Place the CSV file in `data/raw/best_book_ever_dataset.csv`
3. Run the data cleaning notebook to generate processed data

## Usage

### 1. Data Preprocessing

Open and run `notebooks/01_data_cleaning.ipynb`:
- Loads raw Goodreads data
- Cleans text (removes HTML, special characters)
- Extracts and balances genres
- Applies preprocessing (lowercasing, tokenization, lemmatization, stop word removal)
- Saves cleaned dataset to `data/processed/`

### 2. Train Logistic Regression Model

Open and run `notebooks/02_train_logistic_regression.ipynb`:
- Converts text to TF-IDF features
- Trains Logistic Regression classifier
- Evaluates performance with confusion matrices and metrics
- Training time: approximately 2 minutes

### 3. Train DistilBERT Model

Open and run `notebooks/03_train_Distillbert.ipynb`:
- Tokenizes text using DistilBERT tokenizer
- Fine-tunes pre-trained DistilBERT model
- Evaluates with detailed metrics and visualizations
- Training time: approximately 25 minutes (with GPU acceleration)

**Note**: DistilBERT training requires significant computational resources. GPU acceleration is recommended but not required.

## Model Performance

### Overall Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 63.10% | 0.6319 | 0.6310 | 0.6291 |
| DistilBERT | 68.27% | 0.6851 | 0.6827 | 0.6834 |

### Per-Genre Performance (F1-Score)

| Genre | Logistic Regression | DistilBERT | Change |
|-------|---------------------|------------|--------|
| Self-Help | 0.809 | 0.818 | +1.1% |
| Biography | 0.625 | 0.784 | +25.4% |
| Horror | 0.751 | 0.731 | -2.7% |
| Mystery | 0.606 | 0.660 | +8.9% |
| Science Fiction | 0.566 | 0.658 | +16.3% |
| Fantasy | 0.569 | 0.630 | +10.7% |
| Romance | 0.441 | 0.728 | +65.1% |
| Fiction | 0.766 | 0.534 | -30.3% |

### Key Findings

- DistilBERT significantly improved genres requiring contextual understanding (Romance, Biography)
- Both models struggled with overlapping fiction subgenres (Fantasy, Science Fiction, Horror)
- Logistic Regression showed underfitting with consistent 62% performance across cross-validation
- DistilBERT showed overfitting with 92% training accuracy vs 68% test accuracy

## Methodology

### Text Preprocessing
- Lowercasing for consistency
- Tokenization using NLTK
- Removal of punctuation and special characters
- Stop word removal (common words like "the", "and")
- Lemmatization to reduce words to base forms

### Feature Extraction
- **TF-IDF**: Measures word importance relative to corpus
- **DistilBERT Embeddings**: Context-aware semantic representations

### Model Training
- **Logistic Regression**: Trained with L2 regularization
- **DistilBERT**: Fine-tuned for 4 epochs with early stopping consideration

## Technical Challenges and Solutions

### Challenge 1: Missing Genres
- **Problem**: Science Fiction and Self-Help initially absent due to naming variations
- **Solution**: Priority-based extraction with multiple term variations

### Challenge 2: HTML Noise
- **Problem**: Raw descriptions contained HTML tags and entities
- **Solution**: Regex-based cleaning and normalization

### Challenge 3: Multi-Genre Books
- **Problem**: Books belong to multiple genres but needed single labels
- **Solution**: Priority ordering favoring rare genres; acknowledged 10-15% ambiguity

### Challenge 4: Computational Constraints
- **Problem**: DistilBERT memory-intensive on limited hardware
- **Solution**: Optimized batch size (16), token limit (256), GPU acceleration

## Limitations and Future Work

### Current Limitations
- Dataset bias toward mainstream Western literature
- Temporal bias toward recent publications
- Single-label classification simplifies multi-genre reality
- DistilBERT overfitting suggests need for better regularization

### Future Improvements
- Implement early stopping at optimal validation point
- Explore multi-label classification for books spanning multiple genres
- Test larger transformer models (BERT-large, RoBERTa)
- Expand dataset with international and classic literature
- Implement ensemble methods combining both approaches
- Deploy as REST API for real-time predictions

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- pandas, numpy (data processing)
- scikit-learn (traditional ML)
- torch, transformers (deep learning)
- nltk (text preprocessing)
- matplotlib, seaborn (visualization)

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting pull requests.

## License

This project is for educational purposes. Dataset usage should comply with Goodreads terms of service.

## Acknowledgments

- Dataset creators: Lorena Casanova Lozano and Sergio Costa Planells (UOC)
- Goodreads for the original book data
- Hugging Face for DistilBERT implementation

## Contact

For questions or feedback, please open an issue in this repository.
