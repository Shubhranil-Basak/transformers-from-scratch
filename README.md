# Transformer from Scratch

This repository contains an implementation of transformers from scratch using PyTorch. This project follows the "Attention is All You Need" research paper and includes all the necessary components to train and evaluate the model on a dataset from Hugging Face.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project includes the following files in the `transformer` directory:
- `model.py`: Defines the transformer model architecture.
- `train.py`: Contains the training loop and evaluation logic along with code to download the dataset.
- `config.py`: Provides a function to return different hyperparameters for training and the path to weights for the model.
- `dataset.py`: Handles preprocessing of the dataset from Hugging Face.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face `datasets` library
- Other dependencies listed in `requirements.txt`

## Installation

Clone the repository and navigate to the `transformer` directory:

```bash
git clone https://github.com/Shubhranil-Basak/transformers-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

## Usage

### Configuration

You can set different hyperparameters for training by modifying the `config.py` file. The `get_config` function returns a dictionary containing all the hyperparameters. You can also change the source and target language here along with the dataset of your choice. It uses opus_books by default.

### Dataset Preparation

The `dataset.py` file is responsible for preprocessing the dataset. You can modify this file to change the preprocessing steps.

### Training the Model

To train the model, run the `train.py` script:

```bash
python train.py
```

This script will use the configuration from `config.py` and the dataset from `dataset.py` to train the transformer model.

## Results

After training, the results will be saved in a tensorboard file. The tokenizers are saved in the `tokenizer.src_language_code.json` and `tokenizer.tgt_language_code.json` files and the weights are stored in `opus_books_weights` folder.

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://www.youtube.com/watch?v=bCz4OMemCcA)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---
