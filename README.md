# Project README

## 🔄 Reproducibility Instructions

This repository contains all code and scripts required for the project.  
There are **two ways to reproduce the full pipeline**:

### Option 1: Full Reproduction from Scratch
You can fully reproduce the results by running all notebooks in order.  
The notebooks are named with the following format:


> ⚠️ It is important to execute the notebooks strictly according to their numerical order XXXX_<file_num>.ipynb to ensure that data preparation, model training, and evaluation steps execute correctly.

### Option 2: Partial Reproduction (Recommended for Convenience)
To avoid running the entire pipeline from scratch, you can download all pre-generated data, tokenized files, and trained models that are stored externally in Google Drive (see sections below).  
By placing these files into their corresponding local directories, you will be able to:

- Skip data preprocessing and model training.
- Directly run the analysis, evaluation, or inference notebooks.

---

## 📂 Externally Stored Data & Models

The following directories contain files that are not included directly in the repository and are stored externally on Google Drive:

- `data/`
- `Numeric_Features_model/trained-model/`
- `userdesc-LM-model/tokenized_data_train_balanced/`
- `userdesc-LM-model/trained-model/`
- `result_analysis/model_data/`

These files include:

- Processed datasets
- Tokenized input data
- Trained models (both numeric and language-based models)
- Intermediate outputs used in downstream analysis

---

## 🔑 Access Instructions

1. **Request permission** from the project owner to access the relevant Google Drive folders.
2. Once granted, download the files and place them into the corresponding local directories listed above.
3. Ensure that directory names and structures exactly match, otherwise file paths referenced in the code may fail.
4. Link for the Google Drive folder: [Google Drive Link](https://drive.google.com/drive/folders/1ox-ts3BTSjR-zu8UbjQ4-lzvwY0nqoBW?usp=sharing)
> Note: You only need to download these files if you wish to avoid running the full pipeline from scratch.

---

## ⚠️ Important Notes

- Some notebooks and scripts will not function properly if the required data and models are missing from the expected directories.
- This external storage approach keeps the repository lightweight and manageable while preserving full reproducibility for authorized users.

---

