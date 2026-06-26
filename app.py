import streamlit as st
import pickle

st.markdown("""
<style>
.stApp{
    background-color:#E6F0FF;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div.stButton > button {
    background-color:#2563EB;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("models/fake_news_model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# Page title
st.markdown("""
<h1 style='
text-align:center;
color:#1E3A8A;
white-space: nowrap;
font-size:48px;
'>
📰 Fake News Detector for Students
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='
text-align:center;
font-size:22px;
color:#4B5563;
font-weight:600;
margin-bottom:30px;
'>
Analyze news articles and identify misinformation using Machine Learning.
</p>
""", unsafe_allow_html=True)

# User input
news_text = st.text_area(
    "Paste News Article Here",
    height=250,
)

# Prediction button
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        transformed_text = vectorizer.transform([news_text])

        prediction = model.predict(transformed_text)

        probability = model.predict_proba(transformed_text)

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.error("⚠️ Fake News Detected")

            st.warning(
                "This article may contain misinformation. "
                "Please verify the information before sharing."
            )
            
            st.markdown("### Verify Information Using")

            st.info("""
                • BBC News

                • Reuters

                • The Hindu

                • Associated Press
            """)
        else:
            st.success("✅ Real News")

        st.markdown(
f"""
<div style="
background-color:#DBEAFE;
padding:15px;
border-radius:10px;
text-align:center;
font-size:22px;
font-weight:bold;
color:#1E40AF;
margin-top:10px;
">
🎯 Confidence Score: {confidence:.2f}%
</div>
""",
unsafe_allow_html=True
)