# HamiSkills Spam Email Classifier

This project demonstrates a **spam email/SMS classifier** for Hami MiniMarket using Python and scikit-learn. It trains machine learning models to detect whether a message is "Spam" or "Ham" (not spam) and includes a CLI for real-time testing.

---

## Dataset Used

- **SMS Spam Collection Dataset**
- Format: Tab-separated values with columns:
  - `label` → "ham" (not spam) or "spam"
  - `message` → Text message content
- Dataset size: 5,572 messages
  - Ham: 4,825 messages
  - Spam: 747 messages

---

## Steps Taken

1. **Load Libraries**
   - Pandas, NumPy for data handling
   - scikit-learn for ML models and feature extraction
   - Matplotlib and Seaborn for visualization
   - Joblib for saving/loading models

2. **Load and Explore Dataset**
   - Checked for null values
   - Explored class balance

3. **Data Preprocessing**
   - Encode labels (`ham=0`, `spam=1`)
   - Split dataset into training and testing sets

4. **Feature Extraction**
   - Convert text messages to numerical features using **TF-IDF Vectorization** with unigrams and bigrams

5. **Model Training**
   - **Logistic Regression** (with class balancing)
   - **Multinomial Naive Bayes**

6. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion Matrix visualization
   - Logistic Regression achieved ~99% accuracy
   - Naive Bayes achieved ~97% accuracy

7. **Save Models**
   - TF-IDF vectorizer and trained models saved using `joblib`

8. **Prediction on New Messages**
   - Function to predict whether a new message is Spam or Ham
   - Batch testing and CLI interface included

9. **CLI for Real-Time Testing**
   - Users can input messages and receive predictions from both models
   - Type `exit` to quit the CLI

---

## How to Run the Notebook

1. **Clone or Download the Repository**

```bash
git clone <repository-url>
cd HamiSkills-Spam-Classifier
---
2. Install Dependencies
!pip install pandas numpy scikit-learn matplotlib seaborn joblib

3. Run the Notebook
Open HamiSkills_Spam_Classifier.ipynb in Jupyter Notebook or JupyterLab
Execute all cells sequentially

4. Use the CLI
python HamiSkills_Spam_Classifier.ipynb
