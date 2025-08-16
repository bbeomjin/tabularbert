# TabularBERT

TabularBERT is a comprehensive framework for tabular data modeling using BERT-based transformers. This package provides tools for pretraining and fine-tuning BERT models on tabular datasets, enabling powerful representation learning for structured data.

## Features

- BERT-based architecture specifically designed for tabular data
- Support for both pretraining and fine-tuning workflows
- Built-in data preprocessing and encoding utilities

## Installation

### Method 1: Install from Source (Recommended)

1. **Download the package source from GitHub:**
   ```bash
   git clone https://github.com/bbeomjin/tabularbert.git
   ```

2. **Navigate to the package directory:**
   ```bash
   cd tabularbert
   ```

3. **Install the package locally:**
   ```bash
   pip install -e .
   ```

### Method 2: Install from ZIP Archive

1. **Download the ZIP file from GitHub:**
   - Go to https://github.com/bbeomjin/tabularbert
   - Click "Code" → "Download ZIP"

2. **Unzip the package file:**
   ```bash
   unzip tabularbert-main.zip
   cd tabularbert-main
   ```

3. **Install from files locally:**
   ```bash
   pip install -e .
   ```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- Other dependencies (see `requirements.txt`)

## Quick Start

Here's a basic example of how to use TabularBERT:

```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tabularbert import TabularBERTTrainer

# Load your tabular data
data = pd.read_csv("your_dataset.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split and preprocess data
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

# Scale features
scaler = QuantileTransformer(n_quantiles=10000, output_distribution='uniform')
train_X_scaled = scaler.fit_transform(train_X)

# Initialize TabularBERT trainer
trainer = TabularBERTTrainer(
    x=train_X_scaled,
    num_bins=50,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Setup directories and logging
trainer.setup_directories_and_logging(
    save_dir='./pretraining',
    phase='pretraining',
    project_name='My TabularBERT Project'
)

# Start pretraining
trainer.pretrain()
```

## Project Structure

```
tabularbert/
├── tabularbert/          # Main package directory
├── datasets/             # Example datasets
├── pretraining/          # Pretraining scripts and configs
├── fine-tuning/          # Fine-tuning scripts and configs
├── example.py            # Usage example
├── requirements.txt      # Package dependencies
├── setup.py             # Package setup configuration
└── README.md            # This file
```

## Documentation

For more detailed documentation and advanced usage examples, please refer to:
- `example.py` - Complete usage example

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use TabularBERT in your research, please cite:

```bibtex
@misc{tabularbert,
  author = {Beomjin Park},
  title = {TabularBERT: A BERT-based Framework for Tabular Data},
  year = {2025},
  url = {https://github.com/bbeomjin/tabularbert}
}
```

## Contact

- Author: Beomjin Park
- Email: bbeomjin@gmail.com
- GitHub: https://github.com/bbeomjin/tabularbert

