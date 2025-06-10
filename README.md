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

# Create a new conda environment (here named aicellfate):
conda create -n aicellfate python=3.10
conda activate aicellfate

#Install the required packages:
pip install -r requirements.txt

```

## Usage

Open the Jupyter notebook notebooks/AI4CellFate_workflow.ipynb. The notebook contains the complete documented AI4CellFate workflow: Data loading, Model Training and Evaluation and Visualisation. All cells should be ran sequentially.

- Load the data: Under the “Load Data” cell, replace the x_train, y_train, x_test and y_test with your data, and run the cell. For the following cells, it is expected that the images have a shape of (cells, height, width) and the labels have a shape of (cells, ).

- Train the AI4CellFate model (under the "Optional" cell): Training stage 1: train the adversarial autoencoder by running the according cell. Here you can tune hyperparameters such as the number of epochs, batch size, learning rate, and the lambdas (or “weights”) of the reconstruction and adversarial losses. Training stage 2: train the full AI4CellFate model (this includes the latent space engineering). Here, you can also tune hyperparameters as previously, and now including the lambdas (or “weights”) of the contrastive and covariance losses.

- Latent Space Visualisation: after full model training, visualise the Latent Spaces by running the next cells.

- Classify from the Latent Space: run the next cell to train an MLP to predict the individual cell fates. 

- Visual Interpretation: Run the following cells to perform image perturbations on the chosen latent features.


## Results

The training process generates:
- Model weights saved in `./results/models/`
- Evaluation results in `./results/optimisation/`
- Latent space visualizations
- Reconstructed images

