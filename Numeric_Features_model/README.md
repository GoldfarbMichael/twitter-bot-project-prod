# RandomForest&XGboost-inrersection.ipynb
This project performs binary classification on a labeled dataset of users using both XGBoost and Random Forest classifiers. The workflow includes preprocessing, hyperparameter tuning, model evaluation, and conversion of the best model to ONNX format for GPU inference.

## Dataset

- **Input**: `labeled_intersection.csv`
- **Features**: 
  - `followers`
  - `avg_retweetcount`
- **Target**: 
  - `label` (0 or 1)

## Preprocessing

- Dropped non-informative columns (e.g., `userid`)
- Split dataset into train/test sets (80/20 stratified)

## Models & Training

### 1. XGBoost Classifier

- GPU-accelerated training using `tree_method='gpu_hist'`
- Handled class imbalance with `scale_pos_weight`
- Used `RandomizedSearchCV` with 5-fold cross-validation for hyperparameter tuning
- **Best Test Results**:
  - ROC AUC: `0.889`
  - F1-score (class 1): `0.70`
  - Accuracy: `83%`

### 2. Random Forest Classifier

- Balanced class weights
- Hyperparameter tuning via `RandomizedSearchCV` with 5-fold CV
- **Best Test Results**:
  - ROC AUC: `0.905`
  - F1-score (class 1): `0.71`
  - Accuracy: `83%`
- Saved the best model as `random_forest_model.joblib`

## Model Export

- The trained Random Forest model was exported to ONNX format using `skl2onnx`
- Input shape for ONNX model: 2 numerical features
- Output file: `rf_model.onnx`

## Requirements

- Python >= 3.8
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `skl2onnx`

## How to Use

1. Clone repository and install dependencies.
2. Place your dataset in the `data/` directory.
3. Run the notebook or script to train and evaluate models.
4. Use the exported ONNX model for fast GPU inference.

## Performance Summary

| Model       | ROC AUC | Accuracy | F1-score (class 1) |
|-------------|---------|----------|--------------------|
| XGBoost     | 0.889   | 83%      | 0.70               |
| RandomForest| 0.905   | 83%      | 0.71               |

## Notes

- XGBoost deprecated `gpu_hist` in v2.0+, now prefer `device='cuda'`
- Cleaned unused variables and triggered garbage collection post training

# MLP-intersection.ipynb

This project performs binary classification on user metadata using a Multi-Layer Perceptron (MLP) implemented in PyTorch. It uses focal loss for handling class imbalance and SMOTE for synthetic oversampling. Evaluation includes precision-recall balancing and early stopping.

Dataset
Input: labeled_intersection.csv

Features: All numeric metadata columns except userid, label

Target: label (0 or 1)

Preprocessing
Dropped identifier column (userid)

Features were standardized using StandardScaler

### Data was split into:

- 70% train

- 15% validation

- 15% test
Stratified split to preserve label balance.

Applied SMOTE only on the training set to rebalance the classes.

Computed pos_weight for loss functions:
pos_weight = num_neg / num_pos

## Model & Training
MLPClassifier (PyTorch)
Implemented in Numeric_Features_model/MLPClassifier.py

Used Focal Loss with alpha=0.5, gamma=2.0 to handle hard negatives

Early stopping based on validation loss

Optimizer: Adam (lr=1e-3, weight_decay=1e-5)

Used batch size of 128 (configurable)

### Threshold Tuning
Evaluated validation set across thresholds in [0.35, 0.99]

Selected threshold that maximized a custom weighted blend of precision and recall:
score = β·precision + (1−β)·recall
(β=0.5 used by default)

### Results
- Best Threshold: 0.487
- Train/Test Generalization Gap: 0.0123

### Train Set
ROC AUC: 0.8860

BCE Loss: 0.5267

Accuracy: 80.7%

F1-score (class 1): 0.7921

### Test Set
ROC AUC: 0.8865

BCE Loss: 0.5144

Accuracy: 84.2%

F1-score (class 1): 0.6972

Model Export
Trained PyTorch model saved to disk:

### Requirements
Python >= 3.8

pandas, numpy, scikit-learn

torch, torchvision

imblearn (SMOTE)

matplotlib (optional for ROC plots)

### How to Use
1) Clone repository and install dependencies.

2) Place labeled_intersection.csv inside a data/ folder.

3) Run the training script or notebook.

4) Trained model will be saved as mlp_model.pt.

### Performance Summary
Metric	Train	Test
ROC AUC	0.8860	0.8865
Accuracy	80.7%	84.2%
F1-score (class 1)	0.7921	0.6972
BCE Loss	0.5267	0.5144

### Notes
- SMOTE is only applied to the training set to avoid data leakage.

- Focal Loss improves learning from hard examples and reduces bias from dominant class.

- Threshold optimization provides better trade-off control than default 0.5.



# MLP-intersection.ipynb
# Binary Classification with MLP (PyTorch) and SMOTE Oversampling

This project performs binary classification on user metadata using a Multi-Layer Perceptron (MLP) implemented in PyTorch. It uses focal loss for handling class imbalance and SMOTE for synthetic oversampling. Evaluation includes precision-recall balancing and early stopping.

## Dataset

- **Input**: `labeled_intersection.csv`
- **Features**: All numeric metadata columns except `userid`, `label`
- **Target**: `label` (0 or 1)

## Preprocessing

- Dropped identifier column (`userid`)
- Features were standardized using `StandardScaler`
- Data was split into:
  - 70% train
  - 15% validation
  - 15% test  
  Stratified split to preserve label balance.
- Applied **SMOTE** only on the training set to rebalance the classes.
- Computed `pos_weight` for loss functions:  
  `pos_weight = num_neg / num_pos`

## Model & Training

### MLPClassifier (PyTorch)

- Implemented in `Numeric_Features_model/MLPClassifier.py`
- Used **Focal Loss** with `alpha=0.5`, `gamma=2.0` to handle hard negatives
- Early stopping based on validation loss
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-5`)
- Used batch size of 128 (configurable)

### Threshold Tuning

- Evaluated validation set across thresholds in `[0.35, 0.99]`
- Selected threshold that maximized a custom weighted blend of precision and recall:
  `score = β·precision + (1−β)·recall`  
  (β=0.5 used by default)

## Results

 **Best Threshold**: `0.487`  
 **Train/Test Generalization Gap**: `0.0123`

### Train Set

- **ROC AUC**: `0.8860`
- **BCE Loss**: `0.5267`
- **Accuracy**: `80.7%`
- **F1-score (class 1)**: `0.7921`

### Test Set

- **ROC AUC**: `0.8865`
- **BCE Loss**: `0.5144`
- **Accuracy**: `84.2%`
- **F1-score (class 1)**: `0.6972`

## Model Export

- Trained PyTorch model saved to disk:
  ```
  trained-model/mlp_model.pt
  ```

## Requirements

- Python >= 3.8
- `pandas`, `numpy`, `scikit-learn`
- `torch`, `torchvision`
- `imblearn` (`SMOTE`)
- `matplotlib` (optional for ROC plots)

## How to Use

1. Clone repository and install dependencies.
2. Place `labeled_intersection.csv` inside a `data/` folder.
3. Run the training script or notebook.
4. Trained model will be saved as `mlp_model.pt`.

## Performance Summary

| Metric              | Train       | Test        |
|---------------------|-------------|-------------|
| ROC AUC             | 0.8860      | 0.8865      |
| Accuracy            | 80.7%       | 84.2%       |
| F1-score (class 1)  | 0.7921      | 0.6972      |
| BCE Loss            | 0.5267      | 0.5144      |

## Notes

- SMOTE is only applied to the training set to avoid data leakage.
- Focal Loss improves learning from hard examples and reduces bias from dominant class.
- Threshold optimization provides better trade-off control than default `0.5`.


# MLP-sunset-only.ipynb


This file implements a custom PyTorch-based MLP model that supports **feature masking** to handle incomplete input data, with the goal of classifying social media users (e.g., bots vs. humans).

---

##  Features

- Custom `MaskedMLP` architecture supporting per-sample feature masking.
- Dataset preparation with both full and partial feature availability.
- Early stopping during training.
- Precision-recall-weighted threshold selection.
- Evaluation using confusion matrix, classification report, ROC-AUC, and BCE loss.

---

##  Dataset

The model uses two CSV files:

- `full_features.csv`: Contains all features for each user.
- `partial_features_bots.csv`: Contains limited features (e.g., only followers and retweet count).

Each dataset includes:
- `userid`
- `label` (1 for bot, 0 for human)
- Various numeric features (e.g., followers, avg_retweetcount)

---

##  Architecture

```text
MaskedMLP
├── Linear(input_dim → 128) + ReLU
├── Linear(128 → 64) + ReLU
├── Linear(64 → 32) + ReLU
└── Linear(32 → 1)   (no activation)

