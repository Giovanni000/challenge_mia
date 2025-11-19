# 🏴‍☠️ Pirate Pain Challenge - Notebook Startup Guide

This guide explains how to properly start the three main notebooks of the project: `Pirate_pain_prediction - GRU_BI.ipynb`, `Pirate_pain_prediction - GRU_MONO.ipynb` and `Pirate_pain_prediction - HIGHEST_SCORES.ipynb`.

## Prerequisites

### Python Dependencies

All notebooks require the following Python libraries:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install tensorboard
```

Alternatively, if available, install all dependencies from a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Required Data Files

Make sure you have the following files in the `data/` folder:

- `pirate_pain_train.csv` - Training dataset
- `pirate_pain_train_labels.csv` - Training set labels
- `pirate_pain_test.csv` - Test dataset

The directory structure should be:
```
Pirates/
├── data/
│   ├── pirate_pain_train.csv
│   ├── pirate_pain_train_labels.csv
│   └── pirate_pain_test.csv
├── Pirate_pain_prediction - GRU_BI.ipynb
├── Pirate_pain_prediction - GRU_MONO.ipynb
└── Pirate_pain_prediction - HIGHEST_SCORES.ipynb
```

### Execution Environment

- **Google Colab** (recommended for Pirate_pain_prediction - GRU_MONO.ipynb and Pirate_pain_prediction - HIGHEST_SCORES.ipynb)
- Local **Jupyter Notebook/Lab**
- **VS Code** with Jupyter extension

---

## Notebook 1: Pirate_pain_prediction - GRU_BI.ipynb

### Description
Complete notebook for multivariate time series classification. Implements a full pipeline with preprocessing, feature engineering, RNN/GRU/LSTM model training, cross-validation and submission generation.

### Main Features
- Supports execution in Google Colab or local
- Automatically detects environment (Colab vs local)
- Models: Bidirectional and unidirectional GRU
- 5-fold cross-validation
- Ensemble of top 3 models

### Startup Instructions

#### Option A: Google Colab
1. Upload the notebook to Google Colab
2. Mount Google Drive (if data is on Drive):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Modify the `BASE_DIR` path in the configuration cell if necessary
4. Run all cells in order

#### Option B: Local Execution
1. Make sure data files are in the `data/` folder in the working directory
2. The notebook will automatically detect the local environment
3. Verify that the `BASE_DIR` path is correct:
   ```python
   BASE_DIR = Path('/Users/md101ta/Desktop/Pirates')  # Modify if necessary
   ```
4. Run all cells in order

### Important Configuration
- **SEED**: 2024 (for reproducibility)
- **WINDOW_SIZE**: 25
- **WINDOW_STRIDE**: 15
- **BATCH_SIZE**: 128
- **Device**: Automatically detects CPU/GPU

### Generated Output
- Models saved in `outputs/checkpoints/`
- TensorBoard logs in `outputs/logs/`
- Analysis reports in `outputs/reports/`
- Submission files in `outputs/submission_*.csv`

---

## Notebook 2: Pirate_pain_prediction - GRU_MONO.ipynb

### Description
Notebook that uses embeddings for categorical features (n_legs, n_hands, n_eyes, pain_survey). Implements a GRU model with attention mechanism and uses K-fold cross-validation with ensemble.

### Main Features
- Embeddings for categorical features
- GRU model with Attention
- K-fold cross-validation (5 folds)
- Ensemble of 4 models
- Robust Scaling for normalization

### Startup Instructions

#### Initial Setup (Google Colab)
1. **Mount Google Drive** (first cell):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   current_dir = "/content/drive/My Drive/1st Challenge/"
   %cd $current_dir
   ```

2. **Verify data files**:
   Make sure CSV files are in the current directory or modify paths in data loading cells.

#### Hyperparameter Configuration
The notebook uses the following configurations (defined at the beginning):

```python
# Cross-validation
K = 5                   # Number of folds
N_VAL_USERS = 66        # Users for validation

# Training
EPOCHS = 500            # Maximum epochs
PATIENCE = 35           # Early stopping patience

# Architecture
HIDDEN_LAYERS = 2
HIDDEN_SIZE = 128
RNN_TYPE = 'GRU'
BIDIRECTIONAL = False

# Window and Stride
WINDOW_SIZE = 32
STRIDE = 16
BATCH_SIZE = 32
```

#### Execution
1. Run all cells in sequential order
2. Cross-validation will take time (5 folds × ~500 epochs)
3. Models are automatically saved in `models/gru_mono_test_5_removed_joint10/split_X_model.pt`

#### Submission Generation
The notebook generates two types of submissions:
- **Ensemble** (4 models): `submission_ensemble_noJoint10_4mod_k5.csv`
- **Single model** (split_1): `submission_gru5_split1.csv`

### Important Notes
- The notebook is optimized for Google Colab
- Requires GPU for reasonable training times
- Normalization is applied separately for train and test

---

## Notebook 3: Pirate_pain_prediction - HIGHEST_SCORES.ipynb

### Description
Notebook that implements a bidirectional GRU model with overlapping temporal windows. Includes exploratory data analysis and advanced techniques to handle class imbalance.

### Main Features
- Bidirectional GRU
- Window-based approach with stride
- Class weights to handle imbalance
- Exploratory data analysis
- Support for window importance weighting

### Startup Instructions

#### Initial Setup (Google Colab)
1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount("/gdrive")
   current_dir = "/gdrive/My\\ Drive/[2025-2026]\\ AN2DL/Challenge\\ 1"
   %cd $current_dir
   ```

2. **Verify data files**:
   ```python
   os.environ["TRAIN_FILE"] = "pirate_pain_train.csv"
   os.environ["LABEL_FILE"] = "pirate_pain_train_labels.csv"
   os.environ["TEST_FILE"] = "pirate_pain_test.csv"
   ```

#### Hyperparameter Configuration
Main configurations are defined at the beginning of the notebook:

```python
N_VAL_USERS = 80
N_TEST_USERS = 80
WINDOW_SIZE = 32
STRIDE = 16
LEARNING_RATE = 1e-3
EPOCHS = 500
PATIENCE = 50
HIDDEN_LAYERS = 2
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.2
BATCH_SIZE = 64
```

#### Execution
1. Run setup and import cells
2. Run data loading and preprocessing
3. Run exploratory analysis (optional but recommended)
4. Run model training
5. Generate predictions on test set

### Output
- Models saved in `models/`
- TensorBoard logs in `tensorboard/`
- Submission file with predictions

### Important Notes
- The notebook creates a local test set for validation
- Supports window importance weighting (configurable via `IMPORTANT_SPANS` and `IMPORTANT_ALPHAS`)
- Uses class weights with softening factor β

---

## Troubleshooting Common Issues

### Issue: Files not found
**Solution**: Verify that CSV files are in the correct directory and that paths in the notebook are correct.

### Issue: Out of Memory (OOM)
**Solution**: 
- Reduce `BATCH_SIZE`
- Reduce `WINDOW_SIZE` or increase `STRIDE`
- Use Google Colab with GPU (free)

### Issue: Training too slow
**Solution**:
- Make sure to use GPU (check with `torch.cuda.is_available()`)
- Reduce `EPOCHS` or `PATIENCE` for quick tests
- Reduce `K` (number of folds) in cross-validation

### Issue: Import errors
**Solution**: Install all dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tensorboard
```

### Issue: Incorrect Google Drive paths
**Solution**: Modify paths in Drive mount cells to match your directory structure.

---

## Notebook Comparison

| Feature | Giovanni_Passuello | GABRI | Piero_code |
|---------------|-------------------|-------|------------|
| **Model** | GRU BI/UNI | GRU + Attention | Bidirectional GRU |
| **Embedding** | No | Yes | No |
| **Cross-Validation** | 5-fold | 5-fold | Manual split |
| **Ensemble** | No | 4 models | No |
| **Normalization** | Z-score | Robust Scaling | Standard |
| **Environment** | Colab/Local | Colab | Colab |

---

## Quick Start

### To get started quickly:

1. **Choose a notebook** based on your needs
2. **Prepare data**: make sure CSV files are in the `data/` folder
3. **Configure environment**: 
   - For Colab: mount Drive and modify paths
   - For local: verify paths in code
4. **Run in order**: run all cells sequentially
5. **Wait for training**: times vary from 1-6 hours depending on configuration
6. **Get submissions**: CSV files are automatically generated

---

## Final Notes

- All notebooks use **seed** for reproducibility
- Best models are automatically saved
- TensorBoard logs can be viewed during training
- Submissions are automatically generated at the end of training

For questions or issues, consult the comments within each notebook or the project documentation.

---

**Happy training! 🏴‍☠️**
