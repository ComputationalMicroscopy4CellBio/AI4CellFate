# AI4CellFate

AI4CellFate is a deep learning framework for cell fate prediction and interpretation. The project implements an adversarial autoencoder neural network architecture that combines covariance and contrastive loss terms to optimise the model for fate prediction and interpretation.

## Project Structure

```
AI4CellFate/
├── src/
│   ├── models/         # Neural network model architectures
│   ├── preprocessing/  # Data preprocessing utilities
│   ├── training/       # Training scripts and utilities
│   ├── evaluation/     # Model evaluation tools
│   ├── utils.py        # Helper functions
│   ├── config.py       # Configuration settings
│   └── ai4cellfate.py  # Main module
├── notebooks/          # Jupyter notebooks for analysis
└── setup.py           # Package installation configuration
```

## Features

- **Two-Stage Training Pipeline**:
  1. Autoencoder training for feature extraction
  2. AI4CellFate model training with engineered latent space
- **Advanced Architecture**:
  - Adversarial Autoencoder for dimensionality reduction
  - Covariance learning for capturing relationships
  - Contrastive learning for improved representations
- **Comprehensive Evaluation**:
  - Latent space visualization
  - Reconstruction quality assessment
  - Model performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/ComputationalMicroscopy4CellBio/AI4CellFate.git
cd AI4CellFate

# Install the package
pip install -e .
```

## Usage

```python
from src.ai4cellfate import main

# Run the complete training pipeline
main()
```

## Configuration

The model can be configured through the following parameters:

- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `latent_dim`: Dimension of the latent space
- `GaussianNoise_std`: Standard deviation for noise injection
- `lambda_recon`: Reconstruction loss weight
- `lambda_adv`: Adversarial loss weight
- `lambda_cov`: Covariance loss weight
- `lambda_contra`: Contrastive loss weight

## Results

The training process generates:
- Model weights saved in `./results/models/`
- Evaluation results in `./results/optimisation/`
- Latent space visualizations
- Reconstructed images

