import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="IntelliClassify AI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #312e81,
        #4c1d95
    );
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}

/* Animated Background */

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

/* Main Container */

.block-container {
    padding-top: 2rem;
}

/* Title */

.title {
    font-size: 65px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
    text-shadow: 0px 0px 20px #38bdf8;
    margin-bottom: 10px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #e2e8f0;
    margin-bottom: 40px;
}

/* Glass Card */

.glass {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* Text Area */

.stTextArea textarea {
    background: rgba(255,255,255,0.08);
    color: white;
    border-radius: 15px;
    border: 2px solid #38bdf8;
    font-size: 18px;
}

/* Button */

.stButton button {
    width: 100%;
    height: 60px;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #8b5cf6
    );
    color: white;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    transition: 0.3s;
}

/* Button Hover */

.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #7c3aed
    );
}

/* Result Box */

.result-box {
    padding: 30px;
    border-radius: 20px;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #ffffff;
    margin-top: 30px;
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: 0px 0px 25px rgba(56,189,248,0.6);
}

/* Sidebar Cards */

.side-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* Footer */

.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 60px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = pickle.load(
    open("model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

# =========================
# NLP SETUP
# =========================

lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">🧠 IntelliClassify AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced AI-Powered News Document Classifier</div>',
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================

col1, col2 = st.columns([2, 1])

# =========================
# LEFT SIDE
# =========================

with col1:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    user_input = st.text_area(
        "📰 Enter News Article",
        height=320,
        placeholder="Paste your article here..."
    )

    predict_button = st.button(
        "🚀 Predict Category"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RIGHT SIDE
# =========================

with col2:

    st.markdown("""
    <div class="side-card">

    <h3>📌 Categories</h3>

    🌍 World <br><br>

    ⚽ Sports <br><br>

    💼 Business <br><br>

    🤖 Sci/Tech

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="side-card">

    <h3>✨ Features</h3>

    ✅ NLP Processing <br><br>

    ✅ Machine Learning <br><br>

    ✅ TF-IDF Vectorization <br><br>

    ✅ AI Prediction

    </div>
    """, unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================

if predict_button:

    if user_input.strip() == "":

        st.warning(
            "Please enter article text."
        )

    else:

        cleaned = clean_text(user_input)

        vector = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(vector)

        st.markdown(
            f'''
            <div class="result-box">

            🎯 Predicted Category

            <br><br>

            {prediction[0]}

            </div>
            ''',
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================

st.markdown(
    '''
    <div class="footer">

    Built with ❤️ using NLP, Machine Learning & Streamlit

    </div>
    ''',
    unsafe_allow_html=True
)