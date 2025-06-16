# bot_detection_dataset.ipynb

This notebook performs data cleaning and filtering to prepare a refined dataset of bot accounts for downstream analysis:

- Loaded two datasets: the main labeled dataset (`bot_detection_data.csv`) and processed user metadata (`processed_users.csv`).
- Identified **50,000** unique users in the labeled dataset and examined their overlap with the processed user list, finding **6** intersecting users.
- Filtered the dataset to retain only bot-labeled records (`label = 1`) and further removed users that overlapped with the external processed list, leaving **25,014** clean bot entries.
- Renamed relevant columns for consistency (e.g., `User ID` → `userid`, `Retweet Count` → `avg_retweetcount`).
- Retained only the necessary columns for modeling: `userid`, `avg_retweetcount`, `followers`, and `label`.
- Exported the cleaned dataset to `bot_records_filtered.csv` for further use in model training.

This step ensures a clean and deduplicated bot dataset, ready for numerical model ingestion.


# sunset_pre_preprocessing.ipynb

This notebook processes the labeled_sunset.csv dataset, it drops the redundent columns from the dataframe and reduces all records to one record per unique user, add following features:
- total_tweets - tweets per user on twitter
- Followers and Following counts
- avg_retweetcount -
- Avg_words_per_tweet -
- daily_tweet_count
- unique_language_count
- max_tweers_per_hour
- label - 0 for human, 1 for bot

We also combine proc_df with_bot_records_filterd.csv
In this case we will insert 0 to each missing value that is added form bot_records_filtered.csv the purpose is to enlarge the dataset with bot records


- **Dataset Merging**:
  - Loaded the processed user dataset (`processed_users.csv`) and the cleaned bot-only dataset (`bot_records_filtered.csv`).
  - Aligned the feature columns by adding missing ones to the bot dataset with default values (`0`), ensuring compatibility.
  - Concatenated the two DataFrames to expand the training dataset with more bot samples.

- **Feature Alignment (Intersection Only)**:
  - Identified the common features between the user and bot datasets.
  - Created a unified dataset (`labeled_intersection.csv`) with only intersecting features for fair modeling.

- **User Description Labeling**:
  - Loaded labeled account descriptions (`labeled_sunset.csv`), filtered duplicates and missing data, and mapped text labels ("bot"/"human") to numerical values (`1`/`0`).
  - Exported this cleaned dataset to `userdesc_labeled.csv`.

- **Final Merge for Ensemble Modeling**:
  - Merged user descriptions with the intersection dataset (`labeled_intersection.csv`) to produce `intersection_userdesc_labeled.csv`, which is suitable for ensemble model input.

- **Memory Cleanup**:
  - Deleted unused variables to optimize memory usage during notebook execution.
 
#### Output Files:

- `processed_users.csv`: Updated full dataset with bots included.
- `partial_features.csv`: Bot dataset with full feature columns.
- `labeled_intersection.csv`: Merged dataset with intersecting features only.
- `userdesc_labeled.csv`: Cleaned account descriptions with labels.
- `intersection_userdesc_labeled.csv`: Final dataset combining features and descriptions for modeling.

This process ensures a balanced and feature-aligned dataset, ideal for building robust classification models using both structured features and textual descriptions.
# sunset_unlabeled_preprocessing.ipynb

This notebook processes a directory of CSV files containing Twitter user data. It extracts unique user IDs, filters out already-labeled accounts, and aggregates user statistics for further processing or model training.

---

####  Step-by-Step Summary

##### 1. **Scan All Files for Unique User IDs**
- Reads through 290 CSV files in the `input_dir`.
- Extracts all `userid` values.
- Counts how many times each user appears using a `Counter`.
- Outputs a DataFrame of unique user IDs with occurrence counts (`df_unique`).

-  Output:
- `df_unique`: ~2.4 million unique users with a "count" column indicating how often each appears.

---

##### 2. **Filter Out Already-Labeled Users**
- Loads `labeled_intersection.csv` to get a list of already-used/labeled user IDs.
- Removes those users from `df_unique`.
- Result: `df_unique_filtered`, which contains **only unlabeled users**.

---

##### 3. **Aggregate Tweet Activity (e.g., `totaltweets`)**
- Defines a generic function `scan_and_aggregate()` that:
  - Iterates over all files.
  - Aggregates a target column (like `totaltweets`) by `userid` using an aggregation function (e.g., `max`).
- Merges aggregated results with `df_unique_filtered`.

---

##### 4. **Compute Derived Features**
- Computes `avg_retweetcount` by dividing total retweets by the count of appearances per user.
- Drops raw `retweetcount` column after computing average.
- Keeps relevant features like `followers`, `following`, and `acctdesc`.

---

##### 5. **Save Final Output**
- Saves the final filtered DataFrame as `unique_users_no_intersection.csv`.

---

#### Output File
- `unique_users_no_intersection.csv`: Contains user IDs not used in previous modeling, along with their tweet and account metadata, ready for new model training or annotation.

---

#### Key Features in Final Data:
| Column Name         | Description                                         |
|---------------------|-----------------------------------------------------|
| `userid`            | Unique Twitter user ID                              |
| `count`             | Number of files in which the user appears           |
| `totaltweets`       | Maximum total tweet count found across all files    |
| `avg_retweetcount`  | Average retweet count per appearance                |
| `followers`         | Number of followers (from file data)                |
| `following`         | Number of users this account follows                |
| `acctdesc`          | User profile description (text)                     |

---

This notebook is essential for identifying new users to label or include in model training while ensuring there's no overlap with already-labeled data.
# twittbot22_sunset_merge.ipynb
This notebook downloads, restructures, and filters a 1.2M-row Twitter dataset related to the Ukraine-Russia crisis. The goal is to isolate labeled tweets and generate two datasets: one for bots and one for humans.

---

###  1. Dataset Download

- Uses `kagglehub` to download the **Ukraine-Russian Crisis Twitter Dataset** (1.2M rows, 16.9 GB).
- Files are downloaded to:  
  `/root/.cache/kagglehub/datasets/.../versions/510`

---

###  2. Preprocessing

- Moves dataset to: `/content/ukraine_twitter_dataset`
- Organizes all `.csv` files into a new subdirectory:  
  `/content/ukraine_twitter_dataset/510/files`

- Reads a custom `label.csv` file and removes the prefix `"u"` from `id` values to match the dataset format.
- Saves the cleaned labels as `labels.csv`.

---

### 🧪 3. Extract Labeled Records

- Iterates over all `.csv` files in the `files/` directory.
- Filters out rows that contain non-null values in the `label` column.
- Concatenates all labeled rows into a new DataFrame.
- Saves this labeled dataset as:  
  - `labeled_sunset.csv`

---

### 4. Statistics: Labeled Records

- Computes number of unique labeled IDs:
  - **Total:** `104,288` unique labeled users

---

###  5. Split Labeled Data into Bots and Humans

- Filters `labeled_sunset.csv` into:
  - 🧠 `bot.csv` – rows labeled as `'bot'`
  - 🙋 `human.csv` – rows labeled as `'human'`

### Unique user ID counts:
| Label  | Unique IDs |
|--------|------------|
| Bot    | 6,438      |
| Human  | 97,850     |

---

###  Output Files Summary

| File Name             | Description                              |
|----------------------|------------------------------------------|
| `labels.csv`          | Cleaned version of input `label.csv`     |
| `labeled_sunset.csv`  | All records with non-null `label`        |
| `bot.csv`             | All labeled bot accounts                 |
| `human.csv`           | All labeled human accounts               |

---

### Notes

- `DtypeWarning` may appear due to mixed column types — can be resolved with `low_memory=False` or explicit `dtype`.
- Final outputs are useful for bot detection model training and evaluation.

---

This notebook is ideal for curating a clean and structured dataset to train models that distinguish between bot and human behavior on Twitter.
