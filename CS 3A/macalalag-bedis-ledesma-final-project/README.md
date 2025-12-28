# Sentiment Analysis with CNN

## Project Information

- **Course**: CS 3A (Computer Science 3A)
- **Semester**: 2025-2026
- **Authors**: Macalalag, Bedis, Ledesma

## Overview

This project implements a Convolutional Neural Network (CNN) for sentiment analysis on movie reviews using the IMDB dataset. Built with TensorFlow/Keras, it provides a command-line interface for real-time sentiment prediction and includes a Jupyter notebook for model training and evaluation.

## Features

- CNN-based sentiment classification
- Text preprocessing with tokenization and padding
- Command-line prediction interface
- Jupyter notebook for training and analysis
- High accuracy on IMDB dataset

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
   this repo

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB_Dataset.csv` in the project root.

## Usage

### Training
Run the Jupyter notebook `sentiment-analysis.ipynb` to train the model and generate `sentiment_model.h5` and `tokenizer.pkl`.

### Prediction
Use the command-line script:
```bash
python sentiment-analyser.py
```
Enter a review to get sentiment analysis results.

## Model Architecture

- Embedding Layer (vocab_size=20000, embed_dim=128)
- Conv1D Layer (128 filters, kernel_size=3, ReLU)
- GlobalMaxPooling1D
- Dense Layer (128 units, ReLU)
- Dropout (0.5)
- Output Layer (1 unit, Sigmoid)

**Training Config:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch size: 128
- Max epochs: 10

## Performance

- Test Accuracy: ~85-90%
- Includes precision, recall, and F1-score

## Project Files Access

For quick access to the pretrained model and key files, download the project archive from: [Google Drive Link](https://drive.google.com/drive/folders/1LXOD-T-ZwPWJZKWld6ErO0_3sUoaDdT1?usp=sharing) 

The archive includes:
- `sentiment_model.h5` (pretrained CNN model)
- `sentiment-analyser.py` (prediction script)
- `sentiment-analysis.ipynb` (training notebook)
- `app.py` (additional code)
- `requirements.txt` (dependencies)
- `README.md` (this documentation)
- `IMDB_Dataset.csv` (dataset)

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- IMDB Dataset from Kaggle
- TensorFlow/Keras framework
