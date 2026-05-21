import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# DOWNLOAD NLTK
# =========================

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

/* Main App */

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

/* Animated Gradient */

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

/* Floating Particles */

body::before {

    content: "";

    position: fixed;

    width: 200%;

    height: 200%;

    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.08) 2px,
            transparent 2px
        );

    background-size: 50px 50px;

    animation: moveBackground 25s linear infinite;

    top: -50%;

    left: -50%;

    z-index: -1;
}

@keyframes moveBackground {

    0% {
        transform: translate(0,0);
    }

    100% {
        transform: translate(50px,50px);
    }
}

/* Remove Top Space */

.block-container {
    padding-top: 1rem;
}

/* Main Title */

.main-title {

    font-size: 75px;

    font-weight: bold;

    text-align: center;

    color: white;

    text-shadow: 0px 0px 25px #38bdf8;

    margin-bottom: 10px;
}

/* Subtitle */

.subtitle {

    text-align: center;

    font-size: 24px;

    color: #e2e8f0;

    margin-bottom: 30px;
}

/* Developer */

.developer {

    text-align: center;

    font-size: 20px;

    color: #cbd5e1;

    margin-bottom: 40px;
}

/* Hero Card */

.hero-card {

    background: rgba(255,255,255,0.08);

    padding: 30px;

    border-radius: 25px;

    backdrop-filter: blur(10px);

    border: 1px solid rgba(255,255,255,0.15);

    text-align: center;

    margin-bottom: 35px;

    box-shadow: 0px 0px 30px rgba(56,189,248,0.2);
}

/* Glass Effect */

.glass {

    background: rgba(255,255,255,0.08);

    border-radius: 25px;

    padding: 25px;

    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow: 0px 0px 25px rgba(0,0,0,0.2);
}

/* Text Area */

.stTextArea textarea {

    background: rgba(255,255,255,0.08);

    color: white;

    border-radius: 20px;

    border: 2px solid #38bdf8;

    font-size: 18px;
}

/* Button */

.stButton button {

    width: 100%;

    height: 65px;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #8b5cf6
    );

    color: white;

    font-size: 24px;

    font-weight: bold;

    border-radius: 18px;

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

    padding: 35px;

    border-radius: 25px;

    background: rgba(255,255,255,0.12);

    backdrop-filter: blur(15px);

    text-align: center;

    font-size: 34px;

    font-weight: bold;

    color: white;

    margin-top: 35px;

    border: 2px solid rgba(255,255,255,0.2);

    box-shadow: 0px 0px 30px rgba(56,189,248,0.5);
}

/* Side Cards */

.side-card {

    background: rgba(255,255,255,0.08);

    padding: 25px;

    border-radius: 20px;

    margin-bottom: 25px;

    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.12);
}

/* Category Cards */

.category-container {

    display: flex;

    gap: 15px;

    justify-content: center;

    flex-wrap: wrap;

    margin-top: 25px;
}

.category-card {

    padding: 18px 25px;

    border-radius: 18px;

    background: rgba(255,255,255,0.1);

    font-size: 20px;

    font-weight: bold;

    backdrop-filter: blur(10px);

    transition: 0.3s;
}

.category-card:hover {

    transform: scale(1.05);

    background: rgba(255,255,255,0.2);
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
# SIDEBAR
# =========================

st.sidebar.title("🧠 IntelliClassify AI")

st.sidebar.write("Developed by Sindhu ✨")

st.sidebar.info("""
AI Powered News Classifier

Technologies:
- NLP
- TF-IDF
- LinearSVC
- Streamlit
- Machine Learning
""")

# =========================
# HEADER
# =========================

st.markdown(
    """
    <div class="main-title">
    🧠 IntelliClassify AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Advanced AI-Powered News Document Classifier
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="developer">
    Developed by Sindhu ✨
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# HERO SECTION
# =========================

st.markdown(
    """
    <div class="hero-card">

    <h2>🚀 AI Powered News Intelligence</h2>

    <p style='font-size:20px;'>

    Instantly classify news articles using
    Natural Language Processing and
    Machine Learning.

    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# CATEGORY CARDS
# =========================

st.markdown(
    """
    <div class="category-container">

    <div class="category-card">
    🌍 World
    </div>

    <div class="category-card">
    ⚽ Sports
    </div>

    <div class="category-card">
    💼 Business
    </div>

    <div class="category-card">
    🤖 Sci/Tech
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================
# MAIN LAYOUT
# =========================

col1, col2 = st.columns([2,1])

# =========================
# LEFT SIDE
# =========================

with col1:

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "📰 Enter News Article",
        height=320,
        placeholder="Paste your article here..."
    )

    predict_button = st.button(
        "🚀 Predict Category"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================
# RIGHT SIDE
# =========================

with col2:

    st.markdown(
        """
        <div class="side-card">

        <h3>✨ Features</h3>

        ✅ NLP Processing <br><br>

        ✅ Machine Learning <br><br>

        ✅ TF-IDF Vectorization <br><br>

        ✅ AI Classification <br><br>

        ✅ Real-Time Prediction

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-card">

        <h3>📌 Technologies</h3>

        🐍 Python <br><br>

        🤖 Scikit-Learn <br><br>

        🌐 Streamlit <br><br>

        🧠 NLP

        </div>
        """,
        unsafe_allow_html=True
    )

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

        st.balloons()

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
    """
    <div class="footer">

    🔗 GitHub:
    https://github.com/sindhu834

    <br><br>

    Built with ❤️ using NLP, Machine Learning & Streamlit

    </div>
    """,
    unsafe_allow_html=True
)