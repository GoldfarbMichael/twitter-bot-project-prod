import torch
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
import onnxruntime as ort

class BotEnsemble:
    def __init__(self, transformer_model, tokenizer, numeric_model, alpha=0.5, device='cuda'):
        self.transformer = transformer_model.eval().to(device)
        self.tokenizer = tokenizer
        self.numeric_model = numeric_model
        self.alpha = alpha
        self.device = device

        import torch.nn as nn
        from sklearn.base import BaseEstimator

        self.is_torch_model = isinstance(numeric_model, nn.Module)
        self.is_sklearn_model = isinstance(numeric_model, BaseEstimator)
        self.is_onnx_model = isinstance(numeric_model, ort.InferenceSession)

        if self.is_torch_model:
            self.numeric_model = numeric_model.eval().to(device)

    def predict_prob(self, acctdesc, features):
        # === Numeric prediction ===
        if self.is_torch_model:
            feats_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                numeric_prob = torch.sigmoid(self.numeric_model(feats_tensor)).item()

        elif self.is_sklearn_model:
            feature_df = pd.DataFrame([features], columns=["avg_retweetcount", "followers"])
            numeric_prob = self.numeric_model.predict_proba(feature_df)[0][1]

        elif self.is_onnx_model:
            input_name = self.numeric_model.get_inputs()[0].name
            input_array = np.array([features], dtype=np.float32)
            outputs = self.numeric_model.run(None, {input_name: input_array})
            probs = outputs[1][0]
            numeric_prob = probs[1]

        else:
            raise TypeError("Unsupported numeric model type. Must be PyTorch, sklearn, or ONNX.")

        # === Transformer prediction ===
        if acctdesc and not pd.isna(acctdesc) and str(acctdesc).strip():
            inputs = self.tokenizer(
                acctdesc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_token_type_ids=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.transformer(**inputs).logits
                probs = softmax(logits, dim=1)
                transformer_prob = probs[0, 1].item()

            final_prob = self.alpha * transformer_prob + (1 - self.alpha) * numeric_prob
        else:
            final_prob = numeric_prob

        return final_prob

    def predict_label(self, acctdesc, features, threshold=0.5):

        prob = self.predict_prob(acctdesc, features)
        return int(prob > threshold)




