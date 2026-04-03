# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Overview
This project implements a complete **Natural Language Processing (NLP) pipeline** to classify English news articles as **REAL** or **FAKE** using Machine Learning.

The system takes a news **title + body text** as input and returns:
- Predicted label (**REAL / FAKE**)  
- Confidence score  

A web application is also deployed for real-time predictions.

---

## 🚀 Live Demo
🔗 https://z0ha.pythonanywhere.com  

---

## 📂 Dataset
- **Name:** ISOT Fake and Real News Dataset  
- **Source:** Kaggle  
- **Files Used:**
  - `True.csv` → Real news (Reuters)
  - `Fake.csv` → Fake news articles  

### 📊 Sample Dataset
- 100 balanced instances:
  - 50 REAL  
  - 50 FAKE  
- Used as a **toy dataset** for quick training and deployment

---

## 🔄 NLP Pipeline

### 1. Data Collection
- Loaded dataset in Google Colab  
- Added a `Label` column (REAL / FAKE)

### 2. Feature Engineering
- Combined `title` and `text` into:
  News_Content

### 3. Sampling
- Created balanced dataset using:
  - groupby().apply()
  - random_state=42

### 4. Data Preprocessing
Applied:
- Removal of punctuation, numbers, symbols, URLs  
- Lowercasing  
- Stopword removal using NLTK  

---

## 🔢 Train-Test Split
- Training set: **80%**  
- Testing set: **20%**

---

## 🧠 Feature Extraction
- **TF-IDF Vectorizer**
  - max_features = 500

---

## 🤖 Model Training
- **Algorithm:** Logistic Regression  
- Chosen because:
  - Performs well on text classification  
  - Does not assume word independence (unlike Naive Bayes)

---

## 📈 Model Performance

| Metric     | Score |
|-----------|------|
| Accuracy  | 85%  |
| Precision | 0.89 |
| Recall    | 0.85 |
| F1 Score  | 0.85 |

Correct predictions: **17 / 20 test samples**

---

## 💾 Model Saving
Saved using joblib:
- fake_news_lr_model.pkl  
- tfidf_vectorizer.pkl  

---

## 🌐 Web Application
- Built using **Flask**
- Features:
  - Clean and responsive UI  
  - Real-time predictions  
  - Confidence score display  

### ⚙️ Deployment
- Platform: **PythonAnywhere (Free Tier)**  

---

## 🛠️ Tech Stack

| Category     | Tools Used |
|-------------|----------|
| Language     | Python 3 |
| Environment  | Google Colab |
| Libraries    | pandas, scikit-learn, nltk, joblib |
| Visualization| matplotlib, seaborn |
| Web          | Flask |
| Deployment   | PythonAnywhere |

---

## 📌 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git

# Navigate to project folder
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 📖 Future Improvements
- Train on full dataset (not just 100 samples)  
- Try advanced models (SVM, Random Forest, Deep Learning)  
- Improve UI/UX  
- Add API support  

---
