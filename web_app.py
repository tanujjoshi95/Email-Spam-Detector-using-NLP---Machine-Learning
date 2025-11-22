import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Load the model 

model_data=joblib.load("Model.joblib")
Model=model_data["model"]
tfid_vect=model_data["vect"]



def clean_text(text: str):
    clean_text=[]
    ps=PorterStemmer()
    text = str(text).lower()                      # lowercase the text
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove links
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # Deals with the spaces
    text=text.split()
    text=[ps.stem(word) for word in text if not word in stopwords.words('english')]
    text=' '.join(text)
    clean_text.append(text)
    return clean_text

def result(msg:str):
    text=clean_text(msg)
    vect=tfid_vect.transform(text)
    res=Model.predict(vect)[0]
    output=""
    if(res==0):
        output="✅ Looks Good! This is a Genuine Message ."
    else :
        output="🚨 WARNING: High Probability of SPAM! DO NOT click any links."

    return output



# Streamlit 

# st.title("📬 Spam Detector")

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">📬 Email/SMS Spam Detector</h1>
    <p style="text-align:center; color:gray; margin-top:0;">
        NLP + Machine Learning project to classify messages as <b>Spam</b> or <b>Not Spam</b>.
    </p>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([5, 1])

with left:
    st.subheader("🔹 Enter a message")

    msg = st.text_area("Message content:",
                            height=160,
                            placeholder="Type or paste an email here...")
    if st.button(" Analyze message", use_container_width=True):
        if msg.strip():
            label = result(msg)
            st.markdown("---")
            st.markdown("### 🧾 Prediction result")
            st.write("Prediction: ",label)
        else:
            st.warning("Please type a message first.")


with right:
    with st.sidebar:
        st.markdown("## 🧠 About This Project")
        st.write(
            """
    This web application demonstrates an **Email & SMS Spam Detection Model** built using  
    **Natural Language Processing (NLP)** and **Machine Learning**.

    The goal is to classify messages into two categories:

    - **📌 SPAM**
    - **📌 NOT SPAM**

    This is a classic binary text-classification problem.
            """
        )

        st.markdown("### 🎯 What This App Does")
        st.write(
            """
    - Takes an email/SMS message as input  
    - Cleans and preprocesses the text  
    - Converts text into numerical features using **TF-IDF**  
    - Uses a trained ML model to classify the message  
    - Shows prediction probability & detailed breakdown  
            """
        )

        st.markdown("### 🛠 Technologies Used")
        st.write(
            """
    - **Python**
    - **pandas**, **numpy**
    - **NLTK** (stopwords, stemming)
    - **TF-IDF Vectorization**
    - **scikit-learn** (Logistic Regression / Multinomial Naive Bayes)
    - **joblib** for model loading
    - **Streamlit** for UI
            """
        )

        st.markdown("### 🔬 NLP Workflow")
        st.write(
            """
    1. **Text Cleaning**
    - Lowercasing  
    - Removing links  
    - Removing punctuation & special characters  
    - Removing stopwords  
    - Applying stemming  

    2. **Vectorization**
    - TF-IDF converts text → numeric features  

    3. **Model Training**
    - Logistic Regression  
    - Multinomial Naive Bayes  

    4. **Evaluation**
    - Accuracy  
    - Precision & recall  
    - Confusion matrix  
            """
        )

        st.markdown("### 📦 Deployment")
        st.write(
            """
    The trained model and TF-IDF vectorizer  
    are stored using **joblib** and loaded inside this Streamlit app  
    for live predictions.
            """
        )

        st.markdown("---")
        st.caption("Made by Tara · Data Science & Android dev learner")




