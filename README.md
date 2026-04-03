================================================================
  DEPLOYED APPLICATION URL
================================================================

  https://z0ha.pythonanywhere.com

================================================================
  PROJECT SUMMARY — FAKE NEWS DETECTION USING NLP & ML
================================================================

OVERVIEW
--------
This project implements a complete Natural Language Processing (NLP)
pipeline to automatically classify English news articles as either
REAL or FAKE using Machine Learning. The system accepts a news title
and article body text as input and returns a prediction along with
a confidence score.


DATASET
-------
Dataset  : ISOT Fake and Real News Dataset
Source   : Kaggle
Link     : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Files    : True.csv (real news from Reuters) and Fake.csv (fake news websites)
Sample   : 100 balanced instances — 50 REAL and 50 FAKE (toy dataset)


INPUT & OUTPUT
--------------
Input  : English news article (title + body text combined)
Output : Predicted label — REAL or FAKE — with confidence score


PIPELINE STEPS
--------------
1. Data Collection
   Loaded True.csv and Fake.csv from Google Drive in Google Colab.
   Added a Label column (REAL / FAKE) to each dataset.

2. Feature Engineering
   Combined the 'title' and 'text' columns into a single field
   called 'News_Content' to give the model maximum text signal.

3. Dataset Sampling (Toy Dataset)
   Created a balanced sample of 100 instances (50 REAL + 50 FAKE)
   using groupby().apply() with random_state=42 for reproducibility.
   Saved as sample_news.csv.

4. Data Cleaning & Preprocessing
   Applied three preprocessing steps to every article:
     - Removed symbols, numbers, punctuation, and URLs
     - Converted all text to lowercase
     - Removed English stopwords using NLTK
   Saved cleaned data as cleaned_sample_news.csv.

5. Train/Test Split
   Split the 100-instance sample into:
     - 80 rows for training (80%)
     - 20 rows for testing  (20%)

6. Feature Extraction
   Used TF-IDF Vectorizer (max_features=500) to convert cleaned
   text into numerical feature vectors for model input.

7. Model Training
   Trained a Logistic Regression classifier on the TF-IDF feature
   matrix. Logistic Regression was chosen over Naive Bayes because
   it does not assume word independence and performs significantly
   better on news article text classification tasks.

8. Model Evaluation
   Accuracy  : 85%
   Precision : 0.89
   Recall    : 0.85
   F1-Score  : 0.85
   The model correctly classified 17 out of 20 test articles.

9. Model Saving
   Saved the trained model and vectorizer as .pkl files using joblib:
     - fake_news_lr_model.pkl
     - tfidf_vectorizer.pkl

10. Web Application Deployment
    Built a Flask web application with a modern, responsive UI.
    Deployed live on PythonAnywhere (free tier).
    The app accepts user text input, preprocesses it, applies the
    trained model, and returns a REAL or FAKE prediction with
    confidence score in real time.


TECHNOLOGY STACK
----------------
Language   : Python 3.x
IDE        : Google Colab (Jupyter Notebook)
Libraries  : pandas, scikit-learn, nltk, joblib, matplotlib, seaborn
Web        : Flask
Deployment : PythonAnywhere
Algorithm  : Logistic Regression + TF-IDF Vectorizer


LIVE URL
--------
  https://z0ha.pythonanywhere.com

================================================================
