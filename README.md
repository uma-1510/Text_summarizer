## Text Summarizer (Seq2Seq + Attention)

A text summarization project using a Seq2Seq model with attention to generate paper titles from abstracts, trained on ArXiv metadata. This project demonstrates strong understanding of NLP basics, preprocessing, tokenization, sequence modeling, and model deployment.

## Project Overview

Task: Automatic text summarization

Input: Abstracts of scientific papers

Output: Predicted paper title

Model: Sequence-to-Sequence (LSTM) with Attention

Framework: TensorFlow / Keras

Goal: Showcase NLP basics, data preprocessing, and deep learning pipelines for summarization.

## Data Source

We used the arXiv metadata snapshot
 dataset:

Format: JSON, contains metadata of ~1.7 million papers

## Key columns used:

abstract → Input to the model

title → Target for the model

For this project, we limited to first 100,000 records for faster training.

## Note: The dataset JSON file is too large for GitHub. Download it from the Kaggle link above.

## Data Preprocessing

## Cleaning text:

Removed HTML tags using BeautifulSoup

Tokenized text using NLTK

Lowercased and removed stopwords

Kept words with length > 2

Expanded contractions (optional)

## Target preparation:

Added special tokens sos (start of sequence) and eos (end of sequence) to titles

## Splitting data:

Training/Test split: 80/20

## Tokenization & padding:

Input abstracts and target titles were converted to sequences of integers

Padded sequences to fixed lengths:

Abstracts: 300 words (MAX_INPUT_LEN)

Titles: 20 words (MAX_TARGET_LEN)


## Files Generated
✅ Tokenizer Pickle Files

These allow transforming raw text into integer sequences consistently:

in_tokenizer.pkl → Input abstract tokenizer

tr_tokenizer.pkl → Target title tokenizer

✅ Keras Model Files

arxiv_summarizer.keras → Trained Seq2Seq model with attention, ready for inference


 ## Model Architecture

Encoder: 3-layer stacked LSTM, latent dimension = 128

Decoder: 1-layer LSTM, latent dimension = 128

Attention: Standard Keras Attention layer to improve alignment

Output Layer: Dense with softmax over target vocabulary

Loss Function: Sparse Categorical Crossentropy
Optimizer: RMSprop


## Training

Epochs: 5 (for demonstration; can increase for better results)

Batch Size: 16

Validation Split: 10%

Example training metrics after 1 epoch:

loss: 2.43
accuracy: 0.65
val_loss: 2.32
val_accuracy: 0.66


 ## With more epochs and GPU support, you can achieve higher accuracy and lower loss.

## Evaluation

After training, the model can be evaluated with:

BLEU score → Measures n-gram overlap with ground truth titles

ROUGE score → Measures recall-oriented overlap

Human-readable examples: Generate predicted titles for test abstracts to qualitatively assess summarization.

## Example Inference:

Abstract:
"We propose a novel transformer-based model for sequence-to-sequence tasks..."

Predicted Title:
"Novel Transformer Model for Seq2Seq Tasks"

## How to Run

## Clone repo:

git clone https://github.com/uma-1510/Text_summarizer.git
cd Text_summarizer

## Install dependencies:

pip install -r requirements.txt


Download ArXiv JSON dataset and place it in project folder.

## Run training script:

python text_summarizer_arxiv.py


## For inference:

from text_summarizer_arxiv import decode_sequence, in_tokenizer, MAX_INPUT_LEN
seq = in_tokenizer.texts_to_sequences(["Enter abstract here"])
seq = pad_sequences(seq, maxlen=MAX_INPUT_LEN, padding="post")
print(decode_sequence(seq))

## Next Steps / Improvements

Train with full 500k+ dataset for better generalization

Switch to transformer-based models (BERT, T5) for state-of-the-art performance

Deploy with Flask + HTML for a user interface

Add ROUGE/BLEU evaluation pipeline

## Folder Structure
Text_summarizer/
│
├─ arxiv-metadata-oai-snapshot.json  # ArXiv dataset (not on GitHub)
├─ text_summarizer_arxiv.py          # Main Seq2Seq training + inference
├─ arxiv_summarizer.keras            # Saved trained model
├─ in_tokenizer.pkl                  # Input tokenizer
├─ tr_tokenizer.pkl                  # Target tokenizer
├─ requirements.txt                  # Python dependencies
└─ README.md                         # This file

## Key Learning Outcomes

Understand Seq2Seq architecture for NLP

Implement attention mechanism in Keras

Perform text preprocessing (cleaning, tokenization, padding)

Use train/test split and evaluation metrics

Save/load models and tokenizers for production-ready pipelines
