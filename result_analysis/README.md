# bot_tweet_topic_analysis.ipynb
This notebook analyzes **bot-generated tweets** from a labeled Twitter dataset, focusing on sentiment, subjectivity, and thematic distribution. The goal is to understand how bots communicate around various topics and how sentiment varies by topic.

---

##  Data Source

- Input: `labeled_tweets.csv` (from Google Drive)
- Each record includes:
  - `text`: The tweet content
  - `cluster_id`: Clustering label for grouping similar tweets
  - `topic_label`: Labeled topics related to each tweet

 Initial dataset shape: `812,048 rows × 3 columns`

---

###  1. Preprocessing

- Drops tweets with missing text
- Fills missing topic labels with empty strings
- Splits multi-topic tweets into individual topic rows (exploded format)

---

###  2. Sentiment & Subjectivity Analysis

- **Sentiment polarity** computed using `TextBlob`
  - Range: `-1` (very negative) to `+1` (very positive)
- **Subjectivity** score: `0` (objective) to `1` (subjective)
- Tweets are categorized into:
  - `Strong Positive`, `Positive`, `Neutral`, `Negative`, `Strong Negative`

### ➕ Sample output:
| Sentiment | Description        |
|-----------|--------------------|
| > 0.2     | Strong Positive    |
| > 0.05    | Positive           |
| ~ 0       | Neutral            |
| < -0.05   | Negative           |
| < -0.2    | Strong Negative    |

---

###  3. Exploratory Visualizations

###  General Sentiment Distribution
- Count of tweets per sentiment category (all bot tweets)

###  Subjectivity Distribution
- Histogram showing how subjective bot tweets are

---

##  4. Topic-Level Analysis

Focuses on the **top 10 most frequent topics** among bot tweets.

### Analyzed Metrics by Topic:
- **Average sentiment** (Bar plot)
- **Tweet volume** (Count plot)
- **Subjectivity distribution** (Box plot)

 All plots are labeled and rotated for readability.

---

##  Insights You Can Derive
- Which topics bots discuss most often
- How positive/negative their tone is on each topic
- Whether bots are using objective or subjective language

# bot_tweets_analysis.ipynb

This notebook processes bot-generated tweets using state-of-the-art sentence embeddings, dimensionality reduction, and clustering. It then applies a Large Language Model (LLM) via the Together.ai API to **automatically label clusters** with human-readable topic descriptions.

---

## Summary of Pipeline

### 1. Text Cleaning
- Removes URLs, mentions, hashtags, and short tweets
- Converts to lowercase and strips non-alphanumeric characters
- Ensures at least 5 tokens per tweet

```python
clean_tweet("Example tweet with #hashtag and @mention http://link")
```
# ➜ "example tweet with and"

### 2. Sentence Embedding

- Uses `SentenceTransformer` model: `all-MiniLM-L6-v2`  
- Leverages **GPU (`cuda`)** for fast batch encoding  
- **Batch size**: 256  
- **Output**: 384-dimensional embeddings per tweet  
- **Saved to**: `model_data/tweet_embeddings.npy`

---

###  3. Dimensionality Reduction

- **PCA**: Reduces embeddings to 50 dimensions  
- **UMAP**: Further compresses to 5D space for clustering  
  - `n_neighbors=5`, `metric='cosine'`, `n_epochs=200`  
- **Output saved as**: `model_data/umap_embeddings.npy`

---

###  4. Clustering with HDBSCAN

- **Clustering algorithm**: `HDBSCAN`  
- Parameters: `min_cluster_size=30`, `metric='euclidean'`  
- Automatically detects **noise** and **variable-sized clusters**  
- **Labels saved to**: `model_data/cluster_labels.npy`

### Example Cluster Output

| Cluster ID | Count     |
|------------|-----------|
| -1         | 314,670   ← noise/unclustered |
| 4154       | 12,408    |
| 772        | 3,018     |
| 5908       | 2,315     |
| ...        | ...       |

---

##  5. Labeling Clusters using LLM (Together.ai)

- Groups tweets by **cluster ID**
- **Samples up to 20 tweets per cluster**
- Sends them to `mistralai/Mistral-7B-Instruct-v0.2` via Together API
- Receives **human-readable topic labels** (e.g., `Russia-Ukraine War`, `Disinformation`, `EU`)

>  Requires a **Together.ai API Key** entered during runtime


# get_bot_tweets.ipynb


This script processes raw Twitter `.csv` files by:
- Merging them with user-level bot/human labels
- Filtering only bot-labeled tweets
- Appending them to a master file
- Tracking processed files to avoid duplication
- Sorting the final result by user ID

---

### 1. Inputs

- **Labeled Users File**:  
  `../data/unique_users_after_labeling2.csv`  
  > Contains `userid`, `label` (1 for bot, 0 for human)

- **Raw Tweet Files Directory**:    
  > Each file should contain `userid` and `text` columns

---

### 2. Processing Logic

- Loads and keeps track of previously processed files using:

- For each new `.csv`:
1. Reads tweets and merges with labels on `userid`
2. Filters for bot users (`label == 1`)
3. Collects tweets in memory
4. When reaching 10,000+ tweets:
   - Appends them to `../data/bot_tweets_by_user.csv`
   - Records processed filename

- At the end:
- Appends any remaining data
- Sorts final output by `userid`

---

### 3. Output Files

| File | Description |
|------|-------------|
| `../data/bot_tweets_by_user.csv` | All tweets from users labeled as bots |
| `processed_files_report.txt` | Logs filenames already processed |

---

###  Features

-  Efficient batching and file writing
-  Resume-safe: skips already-processed files
-  Handles missing columns and errors gracefully
-  Final output is sorted by `userid` for consistency

---

### Output Summary

- **Total Bot Tweets:** `380,119`
- **Columns:** `userid`, `text`

#### Sample Output

```plaintext
 userid                                               text
0    1968  @VeritasVinnie21 @MrChuckD They traded her fre...
1    1968  @gloria_sin It's not the weapons, its avoiding...
2   59563  Finally!!! #Messi𓃵 ❤️❤️❤️ #WorldCupFinal #Arge...
```

# twitter_conclusions.ipynb


This notebook analyzes user-level features to understand behavioral and textual patterns of bots and humans on Twitter — particularly in the context of the Russia–Ukraine war dataset.

---

##  1. Data Source

- **Input File**:  
  `../data/unique_users_after_labeling2.csv`  

Each row represents a unique Twitter user with the following fields:

| Column Name         | Description                                |
|---------------------|--------------------------------------------|
| `userid`            | Unique Twitter user ID                     |
| `totaltweets`       | Total tweets by the user                   |
| `avg_retweetcount`  | Average retweets per tweet                 |
| `followers`         | Number of followers                        |
| `following`         | Number of accounts followed                |
| `acctdesc`          | User-provided bio/description              |
| `label`             | 0 = Human, 1 = Bot (model-labeled)         |

---

##  Overview of Findings

### Label Distribution

- **Humans**: 2,324,724 users  
- **Bots**: 64,955 users  

> Bots comprise ~2.7% of labeled users.

![Label Distribution](label_distribution.png) <!-- Replace with your image path if exporting -->

---

##  Feature Comparisons

### 1. Behavioral Feature Averages

| Label | Total Tweets | Avg Retweet Count | Followers | Following |
|-------|--------------|-------------------|-----------|-----------|
| Human | 18,722       | 406               | 4,613     | 1,045     |
| Bot   | 12,265       | 301               | 1,568     |   918     |

 **Conclusion**: Humans tend to tweet more, have more followers, and receive more retweets than bots.

---

### 2. Tweet Volume Distribution

>  **Observation**: Bots are concentrated in lower tweet volumes; humans show broader, higher activity levels.

![Tweet Volume](tweet_volume.png) <!-- Replace with your image path if exporting -->

---

### 3. Average Retweet Count (Boxplot)

>  **Conclusion**: Humans tend to achieve higher engagement with more frequent extreme values (viral posts).

![Retweet Boxplot](retweet_boxplot.png)

---

### 4. Followers vs Following (Scatter)

>  **Key Insight**: Bots often have a low follower-to-following ratio — a red flag for automation.

---

### 5. Follower-to-Following Ratio

> Bots typically follow many accounts while receiving relatively few followers — a pattern consistent with inauthentic behavior.

---

### 6. Feature Correlation Heatmap

>  **Conclusion**: Weak feature correlations indicate that each feature adds independent predictive value for classification.

---

##  Textual Analysis: User Descriptions

### Word Cloud of Most Common Terms

- **Top Human Keywords**:  
  `love`, `music`, `life`, `fan`, `proud`, `god`  

- **Top Bot Keywords**:  
  `crypto`, `nft`, `bitcoin`, `trader`, `engineer`, `marketing`  

 **Conclusion**:  
Human bios are emotionally expressive and socially oriented.  
Bots lean toward promotional, technical, or ideological language.

---

##  Summary Statistics Table

Detailed breakdown of mean, median, and standard deviation per label:

| Label | Tweet Mean | Median | Std | Retweet Mean | Median | Std | Followers Mean | Median | Std | Following Mean | Median | Std |
|-------|------------|--------|-----|---------------|--------|-----|----------------|--------|-----|----------------|--------|-----|
| Human | 18722.5    | 3017   | 66831 | 406.1       | 0.67   | 2194 | 4613.8         | 168    | 142167 | 1045.6        | 320    | 4366 |
| Bot   | 12265.8    | 1630   | 39909 | 300.7       | 0.00   | 1863 | 1568.6         | 126    | 17742  | 918.1         | 246    | 3120 |

---

##  Research Conclusions

###  Core Research Question  
**How do bots influence social discourse during the Russia–Ukraine war?**

---

###  Key Findings

- **Bots Are Present but a Minority**  
  Despite their small proportion, bots (~65K users) show significant automated participation.

- **Distinct Behavioral Traits**  
  Bots have fewer followers, follow more accounts, and maintain a lower follower-to-following ratio.

- **Commercial & Ideological Messaging**  
  Bot bios include frequent mentions of crypto, marketing, trading — signaling coordinated intent.

- **Amplification > Engagement**  
  Bots prioritize message dissemination rather than organic interaction.

- **Qualitative Impact**  
  Bots contribute to narrative shaping during high-tension events, even without high engagement.

---

##  Summary

- Majority of accounts were labeled as human (~97%).
- Bots display consistent behavioral and textual distinctions.
- Key features:
  - Lower retweet counts
  - Follower/following imbalance
  - Commercial/ideological account descriptions
- The model successfully distinguishes user types using metadata + text features.
- Useful for scalable bot detection during major global events.




