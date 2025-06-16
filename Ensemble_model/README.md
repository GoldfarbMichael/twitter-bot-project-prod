# ensemble.py

The `BotEnsemble` class combines predictions from two models:
- A **transformer-based text classifier** (e.g. DistilBERT) on the user's bio/description.
- A **numeric metadata model** (e.g. sklearn, PyTorch, or ONNX) using features like retweets and followers.

Final prediction = weighted average of the two models.

---

### Dependencies

```python
import torch
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import onnxruntime as ort
```
Class: BotEnsemble

```python
BotEnsemble(transformer_model, tokenizer, numeric_model, alpha=0.5, device='cuda')
```
### Parameters:
Parameter	Type	Description
transformer_model	torch.nn.Module	Pretrained transformer model (e.g. BERT) for bios
tokenizer	HuggingFace tokenizer	Tokenizer for the transformer model
numeric_model	sklearn / PyTorch / ONNX	Model that predicts from numeric metadata
alpha	float (default=0.5)	Weight given to transformer model in the final prediction
device	'cuda' or 'cpu'	Device to run the models on

### Methods

```python
predict_prob(acctdesc, features)
```
Returns the bot probability as a float between 0 and 1.

```python
prob = model.predict_prob("crypto investor & engineer", [avg_retweetcount, followers])
```
If acctdesc is valid, uses both transformer and numeric model.

Final probability:

```python
final_prob = alpha * transformer_prob + (1 - alpha) * numeric_prob
predict_label(acctdesc, features, threshold=0.5)
```
Returns a binary label: 1 for bot, 0 for human.

```python
label = model.predict_label("crypto investor", [250, 120])
```

Model Type	Description:
- PyTorch	Any torch.nn.Module
- Sklearn	Any BaseEstimator with predict_proba
- ONNX	InferenceSession using ONNX runtime

### Internal Workflow
Numeric Model Prediction

### PyTorch:
```python
torch.sigmoid(model(tensor_input)).item()
```
### Sklearn:

python
Copy
Edit
model.predict_proba(df)[0][1]
ONNX:

```python
outputs = model.run(None, {input_name: np_array})
prob = outputs[1][0][1]
```
Transformer Model Prediction

Uses HuggingFace tokenizer + model

Computes:

```python
softmax(logits, dim=1)[0, 1].item()
```
### Final Output:

- Weighted combination:

```python
final_prob = alpha * transformer_prob + (1 - alpha) * numeric_prob
```
### Error Handling
Raises a TypeError if numeric_model is not one of:

torch.nn.Module

sklearn.BaseEstimator

onnxruntime.InferenceSession

### Example Usage
```python
ensemble = BotEnsemble(
    transformer_model=bert_model,
    tokenizer=bert_tokenizer,
    numeric_model=sklearn_model,
    alpha=0.6,
    device='cuda'
)

acctdesc = "🇷🇺 Patriot, crypto analyst, anti-fake news 💯"
features = [320, 1500]  # [avg_retweetcount, followers]

print(ensemble.predict_prob(acctdesc, features))  # → 0.76
print(ensemble.predict_label(acctdesc, features))  # → 1
```
### Notes
- The ensemble allows combining text and metadata for better classification.

- It's flexible to different numeric model types.

- Useful in bot detection, fake profile classification, and hybrid ML systems.
  


# userdesc-numeric-Ensemble.ipynb
This notebook demonstrates a hybrid approach to bot detection by combining:

A transformer-based language model for processing account descriptions.
A numeric-based model (either MLP or ONNX-based Random Forest) for handling user metadata like follower count and average retweet count.
An ensemble mechanism to combine both predictions.
### Components
1. Transformer Model
Architecture: DistilBERT (fine-tuned for sequence classification)
Input: Account description (acctdesc)
Output: Probability of the account being a bot
2. Numeric Models
Two models are used to analyze user statistics:

MLP Classifier (PyTorch)
Random Forest (ONNX)
Features used:

followers
avg_retweetcount
3. BotEnsemble
A custom ensemble class that combines the transformer output with the numeric model output using a weight alpha.




### Performance Summary
MLP Ensemble:

F1 Score: 0.8054
ROC AUC: 0.9136
Random Forest Ensemble:

F1 Score: 0.8414 (better)
ROC AUC: 0.9871 (significantly better)
→ Random Forest wins, especially in overall ranking ability (AUC).

### Labeling Outcome
Labeled Users: 2,285,391
Bots (1): 58,517
Humans (0): 2,226,874
→ Bot rate ≈ 2.56%, which is plausible depending on your domain.

🔍 Suggested Improvements
Parallelize Prediction
You're using apply() with a lambda, which can be slow on millions of users. Consider:

Batch-predicting with your ensemble.
Using joblib.Parallel or multiprocessing for chunk-level parallelization.
If ensemble2 supports batching, rewrite to use ensemble2.predict_proba_batch(...).
Save Intermediate Results
During batch labeling, write a separate CSV per chunk (e.g., batch_00.csv) and combine later. This:

Reduces risk of data loss if interrupted.
Speeds up I/O.
Balance Check
You might want to stratify future sampling for training/validation if retraining based on this full dataset.

Use Stratified Metrics
In addition to macro avg F1/ROC, consider:

Precision@k (e.g., top 1% most bot-likely)
False Positive Rate, especially since humans dominate (class imbalance)
### Final Output


unique_users_after_labeling2.csv

Then aligning with processed_users.csv + userdesc_labeled.csv to build a consolidated dataset.
