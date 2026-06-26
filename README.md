# 📰 Fake News Detector

An interactive web application designed to help students, educators, and researchers detect misinformation in news articles using Machine Learning. The application utilizes a natural language processing (NLP) pipeline paired with a classification model to analyze text and predict whether an article is **Real** or **Fake**, along with a confidence score.

---

## 🚀 Features

- **Real-Time Classification**: Paste any news article content and get immediate classification feedback.
- **Confidence Score**: Displays the model's confidence probability in its prediction.
- **Verification Directory**: If an article is flagged as potentially fake, the app directs users to verified fact-checking and news sources (e.g., BBC News, Reuters, Associated Press, The Hindu).
- **Premium UI**: Designed with a clean, responsive, student-friendly interface built on Streamlit.

---

## 📊 Model Performance & Details

The machine learning pipeline is documented in the [Jupyter Notebooks](models/fnd_model_training_&_evaluation.ipynb) and consists of:
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer with English stop-words filtering and a `max_df` threshold of `0.7`.
- **Classification Model**: Logistic Regression classifier trained on a dataset of **44,898** labeled news articles.

### Performance Metrics (on Test Set)
- **Accuracy**: `98.29%`
- **Precision**: `98.55%`
- **Recall**: `98.17%`
- **F1 Score**: `98.36%`

---

## 📁 Project Structure

```text
├── data/                                 # Raw datasets (Git ignored)
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── fake_news_model.pkl               # Trained logistic regression model (Git ignored)
│   ├── tfidf_vectorizer.pkl              # Fitted TF-IDF Vectorizer (Git ignored)
│   └── fnd_model_training_&_evaluation.ipynb # Model training & evaluation notebook
├── processed_data/
│   ├── cleaned_news.csv                  # Preprocessed dataset (Git ignored)
│   └── fake_news_preprocessing.ipynb     # Preprocessing notebook
├── .gitignore                            # Excludes large CSV and PKL files
├── app.py                                # Streamlit web application
└── requirements.txt                      # Project dependencies
```

> [!NOTE]
> The raw datasets (`data/`) and serialized model files (`models/*.pkl`) are excluded from this GitHub repository using `.gitignore` due to file size limits.

---

## 🛠️ Local Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/SaiSharshitha/Fake-New-Detector.git
cd Fake-New-Detector
```

### 2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
Make sure you have trained/placed the model and vectorizer PKL files in the `models/` directory, then run:
```bash
streamlit run app.py
```
This will launch the app in your default browser at `http://localhost:8501`.
