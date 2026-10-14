import os
import json
import re
import joblib
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

EN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_model.pkl"
)

EN_VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

EN_METRICS_PATH = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

BN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bangla_fake_news_model.pkl"
)

BN_VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "bangla_tfidf_vectorizer.pkl"
)

BN_METRICS_PATH = os.path.join(
    MODEL_DIR,
    "bangla_metrics.json"
)

BN_THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "bangla_threshold.json"
)


# =========================================================
# CSS
# =========================================================
st.html("""
<style>

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 12% 0%,
            rgba(155, 0, 28, 0.20),
            transparent 27%
        ),
        radial-gradient(
            circle at 90% 5%,
            rgba(105, 0, 20, 0.18),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            #020202 0%,
            #080202 48%,
            #140203 100%
        );

    color: #ffffff;
}

.block-container {
    max-width: 1240px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}


/* =====================================================
   PAGE TRANSITION
===================================================== */

.page-enter {
    animation:
        pageEnter
        0.75s
        cubic-bezier(.2,.8,.2,1);
}

@keyframes pageEnter {

    from {
        opacity: 0;

        transform:
            translateY(26px)
            scale(0.975);

        filter: blur(8px);
    }

    to {
        opacity: 1;

        transform:
            translateY(0)
            scale(1);

        filter: blur(0);
    }
}


/* =====================================================
   HOME HERO
===================================================== */

.home-hero {
    position: relative;

    min-height: 520px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    padding: 70px 65px;

    border-radius: 34px;

    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            rgba(13, 13, 13, 0.98),
            rgba(54, 3, 9, 0.93)
        );

    border:
        1px solid
        rgba(255, 65, 85, 0.16);

    box-shadow:
        0 35px 90px
        rgba(0, 0, 0, 0.55);
}

.home-hero::before {
    content: "";

    position: absolute;

    width: 650px;
    height: 650px;

    border-radius: 50%;

    right: -250px;
    top: -330px;

    background:
        radial-gradient(
            circle,
            rgba(210, 8, 45, 0.31),
            transparent 68%
        );

    filter: blur(8px);
}

.home-hero::after {
    content: "";

    position: absolute;

    width: 400px;
    height: 400px;

    border-radius: 50%;

    left: -230px;
    bottom: -250px;

    background:
        radial-gradient(
            circle,
            rgba(120, 0, 20, 0.18),
            transparent 70%
        );
}

.hero-kicker {
    position: relative;
    z-index: 2;

    display: inline-flex;

    align-items: center;

    width: fit-content;

    gap: 10px;

    padding: 8px 15px;

    margin-bottom: 25px;

    border-radius: 999px;

    background:
        rgba(185, 15, 40, 0.08);

    border:
        1px solid
        rgba(255, 80, 100, 0.16);

    color: #c67d87;

    font-size: 11px;

    letter-spacing: 1.9px;

    text-transform: uppercase;

    font-weight: 700;
}

.hero-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #ca1932;

    box-shadow:
        0 0 14px
        rgba(210, 25, 50, 0.9);
}

.home-title {
    position: relative;
    z-index: 2;

    max-width: 850px;

    color: #ffffff;

    font-size:
        clamp(
            55px,
            8vw,
            92px
        );

    line-height: 0.93;

    letter-spacing: -4px;

    font-weight: 820;

    margin-bottom: 28px;
}

.title-accent {
    background:
        linear-gradient(
            90deg,
            #ffffff,
            #efb5bd,
            #b3152d
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-clip: text;
}

.home-description {
    position: relative;
    z-index: 2;

    max-width: 720px;

    color: #9b9799;

    line-height: 1.8;

    font-size: 16px;
}


/* =====================================================
   TITLES
===================================================== */

.eyebrow {
    color: #a34c59;

    font-size: 10px;

    letter-spacing: 2.3px;

    text-transform: uppercase;

    font-weight: 800;

    margin-bottom: 9px;
}

.section-title {
    color: white;

    font-size: 32px;

    font-weight: 760;

    letter-spacing: -0.8px;

    margin-bottom: 20px;
}

.page-title {
    color: white;

    font-size:
        clamp(
            46px,
            6vw,
            70px
        );

    font-weight: 800;

    letter-spacing: -3px;

    line-height: 1;

    margin-bottom: 18px;
}

.page-description {
    max-width: 750px;

    color: #999497;

    font-size: 15px;

    line-height: 1.8;
}


/* =====================================================
   BUTTONS
===================================================== */

div.stButton > button {
    width: 100%;

    min-height: 52px;

    border-radius: 14px;

    border:
        1px solid
        rgba(255, 70, 90, 0.16);

    background:
        linear-gradient(
            135deg,
            #151515,
            #300509
        );

    color: white;

    font-size: 14px;

    font-weight: 700;

    transition:
        all
        0.25s
        ease;
}

div.stButton > button:hover {
    transform:
        translateY(-3px);

    border-color:
        rgba(255, 75, 95, 0.42);

    background:
        linear-gradient(
            135deg,
            #1d1d1d,
            #4b080f
        );

    box-shadow:
        0 14px 34px
        rgba(0, 0, 0, 0.38),
        0 0 25px
        rgba(200, 10, 40, 0.10);
}

div.stButton > button[kind="primary"] {
    min-height: 62px;

    border: none;

    background:
        linear-gradient(
            110deg,
            #7f0c1b,
            #ca1d36,
            #7d0918
        );

    background-size: 200% auto;

    color: white;

    font-size: 15px;

    letter-spacing: 0.2px;

    animation:
        gradientMove
        5s
        ease
        infinite;

    box-shadow:
        0 16px 38px
        rgba(190, 15, 42, 0.22);
}

@keyframes gradientMove {

    0% {
        background-position:
            0%
            center;
    }

    50% {
        background-position:
            100%
            center;
    }

    100% {
        background-position:
            0%
            center;
    }
}


/* =====================================================
   HOME INFO CARDS
===================================================== */

.home-card {
    min-height: 170px;

    padding: 25px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(16, 16, 16, 0.96),
            rgba(40, 4, 8, 0.90)
        );

    border:
        1px solid
        rgba(255, 70, 90, 0.10);

    box-shadow:
        0 15px 35px
        rgba(0, 0, 0, 0.28);

    transition:
        all
        0.25s
        ease;
}

.home-card:hover {
    transform:
        translateY(-5px);

    border-color:
        rgba(255, 70, 90, 0.24);
}

.home-card-title {
    color: white;

    font-size: 18px;

    font-weight: 720;

    margin-bottom: 12px;
}

.home-card-text {
    color: #8c878a;

    font-size: 13px;

    line-height: 1.7;
}


/* =====================================================
   ANALYSIS HEADER
===================================================== */

.analysis-header {
    position: relative;

    overflow: hidden;

    padding: 45px;

    margin-bottom: 35px;

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(15, 15, 15, 0.98),
            rgba(48, 3, 8, 0.92)
        );

    border:
        1px solid
        rgba(255, 70, 90, 0.14);

    box-shadow:
        0 25px 70px
        rgba(0, 0, 0, 0.44);
}


/* =====================================================
   INPUT
===================================================== */

div[data-baseweb="input"],
div[data-baseweb="textarea"] {
    border-radius: 16px !important;

    background:
        rgba(17, 17, 17, 0.94) !important;
}

input,
textarea {
    color: white !important;

    font-size: 15px !important;
}

textarea {
    line-height: 1.7 !important;
}

label {
    color: #bbb6b8 !important;

    font-weight: 600 !important;
}


/* =====================================================
   METRICS
===================================================== */

div[data-testid="stMetric"] {
    min-height: 118px;

    padding: 21px;

    border-radius: 21px;

    background:
        linear-gradient(
            145deg,
            rgba(16, 16, 16, 0.96),
            rgba(40, 4, 8, 0.91)
        );

    border:
        1px solid
        rgba(255, 75, 95, 0.11);

    box-shadow:
        0 14px 34px
        rgba(0, 0, 0, 0.28);
}

div[data-testid="stMetricLabel"] {
    color: #858083;
}

div[data-testid="stMetricValue"] {
    color: white;

    font-weight: 750;
}


/* =====================================================
   RESULT
===================================================== */

.result-shell {
    position: relative;

    overflow: hidden;

    padding: 55px;

    border-radius: 32px;

    background:
        linear-gradient(
            145deg,
            rgba(15, 15, 15, 0.98),
            rgba(52, 4, 10, 0.93)
        );

    border:
        1px solid
        rgba(255, 70, 90, 0.15);

    box-shadow:
        0 35px 90px
        rgba(0, 0, 0, 0.52);
}

.result-shell::before {
    content: "";

    position: absolute;

    width: 530px;
    height: 530px;

    right: -210px;
    top: -290px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(200, 10, 40, 0.28),
            transparent 68%
        );
}

.result-label {
    position: relative;
    z-index: 2;

    color: #a34e5a;

    font-size: 10px;

    letter-spacing: 2.3px;

    text-transform: uppercase;

    font-weight: 800;

    margin-bottom: 12px;
}

.result-title {
    position: relative;
    z-index: 2;

    color: white;

    font-size:
        clamp(
            48px,
            7vw,
            78px
        );

    font-weight: 820;

    letter-spacing: -3.5px;

    line-height: 0.96;

    margin-bottom: 20px;
}

.result-description {
    position: relative;
    z-index: 2;

    color: #979294;

    line-height: 1.8;

    font-size: 15px;

    max-width: 760px;
}

.classification-card {
    margin-top: 30px;

    padding: 36px;

    border-radius: 25px;

    background:
        linear-gradient(
            145deg,
            rgba(18, 18, 18, 0.96),
            rgba(55, 4, 10, 0.89)
        );

    border:
        1px solid
        rgba(255, 75, 95, 0.15);

    box-shadow:
        0 20px 48px
        rgba(0, 0, 0, 0.33);
}

.class-small {
    color: #898487;

    text-transform: uppercase;

    letter-spacing: 2px;

    font-size: 10px;

    margin-bottom: 9px;
}

.class-real {
    color: #d7f2e1;

    font-size: 43px;

    font-weight: 820;
}

.class-fake {
    color: #ffd5da;

    font-size: 43px;

    font-weight: 820;
}

.language-badge {
    display: inline-block;

    margin-top: 15px;

    padding: 8px 13px;

    border-radius: 999px;

    background:
        rgba(255, 255, 255, 0.035);

    border:
        1px solid
        rgba(255, 255, 255, 0.07);

    color: #aaa5a7;

    font-size: 12px;
}


/* =====================================================
   NOTICE
===================================================== */

.notice {
    margin-top: 22px;

    padding: 18px 20px;

    border-radius: 16px;

    background:
        rgba(255, 255, 255, 0.025);

    border:
        1px solid
        rgba(255, 255, 255, 0.055);

    color: #817c7e;

    line-height: 1.7;

    font-size: 13px;
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    margin-top: 60px;

    padding-top: 24px;

    border-top:
        1px solid
        rgba(255, 255, 255, 0.05);

    text-align: center;

    color: #555054;

    font-size: 11px;

    letter-spacing: 0.8px;
}


/* =====================================================
   MOBILE
===================================================== */

@media only screen and (max-width: 768px) {

    .home-hero {
        min-height: 430px;

        padding:
            42px
            28px;
    }

    .home-title {
        font-size: 50px;

        letter-spacing: -2px;
    }

    .analysis-header,
    .result-shell {
        padding:
            36px
            26px;
    }
}

</style>
""")


# =========================================================
# LOAD HELPERS
# =========================================================
@st.cache_resource
def load_pickle(path):

    if not os.path.exists(path):
        return None

    return joblib.load(path)


@st.cache_data
def load_json(path):

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# =========================================================
# LOAD MODELS
# =========================================================
english_model = load_pickle(
    EN_MODEL_PATH
)

english_vectorizer = load_pickle(
    EN_VECTORIZER_PATH
)

english_metrics = load_json(
    EN_METRICS_PATH
)

bangla_model = load_pickle(
    BN_MODEL_PATH
)

bangla_vectorizer = load_pickle(
    BN_VECTORIZER_PATH
)

bangla_metrics = load_json(
    BN_METRICS_PATH
)

threshold_data = load_json(
    BN_THRESHOLD_PATH
)

BANGLA_THRESHOLD = float(
    threshold_data.get(
        "real_threshold",
        0.50
    )
)


# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:

    st.session_state.page = "home"


if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


if "saved_headline" not in st.session_state:

    st.session_state.saved_headline = ""


if "saved_article" not in st.session_state:

    st.session_state.saved_article = ""


# =========================================================
# LANGUAGE DETECTION
# =========================================================
def detect_language(text):

    bangla_chars = re.findall(
        r"[\u0980-\u09FF]",
        text
    )

    english_chars = re.findall(
        r"[A-Za-z]",
        text
    )

    bangla_count = len(
        bangla_chars
    )

    english_count = len(
        english_chars
    )

    total = (
        bangla_count
        + english_count
    )

    if total == 0:

        return "English"

    bangla_ratio = (
        bangla_count
        / total
    )

    if bangla_ratio >= 0.35:

        return "Bangla"

    return "English"


# =========================================================
# GET PROBABILITIES
# =========================================================
def get_probabilities(
    model,
    transformed
):

    probabilities = (
        model.predict_proba(
            transformed
        )[0]
    )

    classes = list(
        model.classes_
    )

    fake_index = classes.index(0)
    real_index = classes.index(1)

    fake_probability = float(
        probabilities[fake_index]
    )

    real_probability = float(
        probabilities[real_index]
    )

    return (
        fake_probability,
        real_probability
    )


# =========================================================
# ANALYSIS FUNCTION
# =========================================================
def analyze_article(text):

    language = detect_language(
        text
    )


    # BANGLA
    if language == "Bangla":

        if (
            bangla_model is None
            or bangla_vectorizer is None
        ):

            return {
                "error":
                "Bangla model files were not found."
            }


        transformed = (
            bangla_vectorizer.transform(
                [text]
            )
        )


        fake_raw, real_raw = (
            get_probabilities(
                bangla_model,
                transformed
            )
        )


        threshold = (
            BANGLA_THRESHOLD
        )


        prediction = (
            1
            if real_raw >= threshold
            else 0
        )


        metrics = (
            bangla_metrics
        )


    # ENGLISH
    else:

        if (
            english_model is None
            or english_vectorizer is None
        ):

            return {
                "error":
                "English model files were not found."
            }


        transformed = (
            english_vectorizer.transform(
                [text]
            )
        )


        fake_raw, real_raw = (
            get_probabilities(
                english_model,
                transformed
            )
        )


        threshold = 0.50


        prediction = (
            1
            if real_raw >= threshold
            else 0
        )


        metrics = (
            english_metrics
        )


    fake_score = (
        fake_raw * 100
    )

    real_score = (
        real_raw * 100
    )


    if prediction == 1:

        classification = (
            "Likely Real"
        )

        model_score = (
            real_score
        )

    else:

        classification = (
            "Likely Fake"
        )

        model_score = (
            fake_score
        )


    return {
        "language":
            language,

        "prediction":
            int(prediction),

        "classification":
            classification,

        "model_score":
            model_score,

        "fake_score":
            fake_score,

        "real_score":
            real_score,

        "threshold":
            threshold,

        "metrics":
            metrics
    }


# =========================================================
# HOME PAGE
# =========================================================
def render_home():

    st.html("""
    <div class="page-enter">

        <div class="home-hero">

            <div class="hero-kicker">
                <span class="hero-dot"></span>
                ML / NLP Intelligence
            </div>

            <div class="home-title">

                Fake News
                <span class="title-accent">
                    Detection
                </span>

            </div>

            <div class="home-description">

                A bilingual Machine Learning system for
                English and Bangla news classification.

                Explore the project or enter the analysis
                environment to evaluate a news article.

            </div>

        </div>

    </div>
    """)


    st.write("")


    # MAIN ANALYSIS BUTTON
    center_left, center, center_right = (
        st.columns(
            [1.2, 2, 1.2]
        )
    )


    with center:

        if st.button(
            "Enter Analysis",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = (
                "analyzer"
            )

            st.rerun()


    st.write("")
    st.write("")


    # INFORMATION CARDS
    card1, card2, card3 = st.columns(3)


    with card1:

        st.html("""
        <div class="home-card">

            <div class="home-card-title">
                Dual Language
            </div>

            <div class="home-card-text">

                Automatically detects whether
                the submitted article is written
                in English or Bangla.

            </div>

        </div>
        """)


    with card2:

        st.html("""
        <div class="home-card">

            <div class="home-card-title">
                Machine Learning
            </div>

            <div class="home-card-text">

                Uses TF-IDF text features and
                Logistic Regression models for
                language-specific classification.

            </div>

        </div>
        """)


    with card3:

        st.html("""
        <div class="home-card">

            <div class="home-card-title">
                Evaluation
            </div>

            <div class="home-card-text">

                Displays model scores,
                classification output and
                evaluation metrics after analysis.

            </div>

        </div>
        """)


    st.html("""
    <div class="footer">
        FAKE NEWS DETECTION
        &nbsp;&nbsp; / &nbsp;&nbsp;
        ENGLISH + BANGLA
        &nbsp;&nbsp; / &nbsp;&nbsp;
        MACHINE LEARNING + NLP
    </div>
    """)


# =========================================================
# ANALYSIS PAGE
# =========================================================
def render_analyzer():

    st.html("""
    <div class="page-enter">

        <div class="analysis-header">

            <div class="eyebrow">
                Analysis Environment
            </div>

            <div class="page-title">
                Analyze News
            </div>

            <div class="page-description">

                Enter a headline and article text below.

                The system will automatically detect
                English or Bangla and select the
                corresponding Machine Learning model.

            </div>

        </div>

    </div>
    """)


    # BACK HOME
    back_col, empty_col = st.columns(
        [1, 4]
    )


    with back_col:

        if st.button(
            "Back to Home",
            use_container_width=True
        ):

            st.session_state.page = (
                "home"
            )

            st.rerun()


    st.write("")
    st.write("")


    input_left, input_right = st.columns(
        [1.55, 0.8],
        gap="large"
    )


    # INPUT FORM
    with input_left:

        st.html("""
        <div class="eyebrow">
            Article Input
        </div>

        <div class="section-title">
            News Content
        </div>
        """)


        headline = st.text_input(
            "Headline",
            value=(
                st.session_state.saved_headline
            ),
            placeholder=(
                "Enter the news headline"
            )
        )


        article = st.text_area(
            "Article Text",
            value=(
                st.session_state.saved_article
            ),
            height=380,
            placeholder=(
                "Paste the complete English "
                "or Bangla news article here."
            )
        )


        analyze_button = st.button(
            "Analyze Article",
            type="primary",
            use_container_width=True
        )


        if analyze_button:

            combined_text = (
                headline.strip()
                + " "
                + article.strip()
            ).strip()


            if len(
                combined_text
            ) < 30:

                st.warning(
                    "Please provide a longer "
                    "headline or article."
                )


            else:

                result = analyze_article(
                    combined_text
                )


                if "error" in result:

                    st.error(
                        result["error"]
                    )


                else:

                    st.session_state.saved_headline = (
                        headline
                    )

                    st.session_state.saved_article = (
                        article
                    )

                    st.session_state.analysis_result = (
                        result
                    )

                    st.session_state.page = (
                        "result"
                    )

                    st.rerun()


    # INFORMATION
    with input_right:

        st.html("""
        <div class="eyebrow">
            Processing Flow
        </div>

        <div class="section-title">
            How It Works
        </div>

        <div class="home-card">

            <div class="home-card-title">
                01 — Detect Language
            </div>

            <div class="home-card-text">

                The system identifies whether
                the input is English or Bangla.

            </div>

        </div>

        <br>

        <div class="home-card">

            <div class="home-card-title">
                02 — Select Model
            </div>

            <div class="home-card-text">

                The corresponding language-specific
                TF-IDF vectorizer and model are loaded.

            </div>

        </div>

        <br>

        <div class="home-card">

            <div class="home-card-title">
                03 — Classify
            </div>

            <div class="home-card-text">

                The article is classified as
                Likely Real or Likely Fake.

            </div>

        </div>
        """)


    st.html("""
    <div class="footer">
        ANALYSIS ENVIRONMENT
        &nbsp;&nbsp; / &nbsp;&nbsp;
        LANGUAGE-AWARE CLASSIFICATION
    </div>
    """)


# =========================================================
# RESULT PAGE
# =========================================================
def render_result():

    result = (
        st.session_state.analysis_result
    )


    if result is None:

        st.session_state.page = (
            "analyzer"
        )

        st.rerun()


    language = result[
        "language"
    ]

    prediction = result[
        "prediction"
    ]

    classification = result[
        "classification"
    ]

    model_score = result[
        "model_score"
    ]

    fake_score = result[
        "fake_score"
    ]

    real_score = result[
        "real_score"
    ]

    threshold = result[
        "threshold"
    ]

    metrics = result[
        "metrics"
    ]


    st.html(
        f"""
        <div class="page-enter">

            <div class="result-shell">

                <div class="result-label">
                    Analysis Complete
                </div>

                <div class="result-title">
                    Article Analysis
                </div>

                <div class="result-description">

                    Detected language:
                    <strong style="color:#ddd7d9;">
                        {language}
                    </strong>

                    <br><br>

                    The article has been processed
                    using the corresponding
                    language-specific classification model.

                </div>

            </div>

        </div>
        """
    )


    class_name = (
        "class-real"
        if prediction == 1
        else "class-fake"
    )


    st.html(
        f"""
        <div class="classification-card">

            <div class="class-small">
                Model Classification
            </div>

            <div class="{class_name}">
                {classification}
            </div>

            <div class="language-badge">
                {language}
            </div>

        </div>
        """
    )


    st.write("")


    score1, score2, score3 = (
        st.columns(3)
    )


    with score1:

        st.metric(
            "Model Score",
            f"{model_score:.2f}%"
        )


    with score2:

        st.metric(
            "Fake Score",
            f"{fake_score:.2f}%"
        )


    with score3:

        st.metric(
            "Real Score",
            f"{real_score:.2f}%"
        )


    st.progress(
        min(
            max(
                model_score / 100,
                0
            ),
            1
        )
    )


    st.write("")
    st.write("")


    st.html("""
    <div class="eyebrow">
        Model Evaluation
    </div>

    <div class="section-title">
        Performance Context
    </div>
    """)


    accuracy = metrics.get(
        "accuracy",
        0
    )

    precision = metrics.get(
        "precision",
        0
    )

    recall = metrics.get(
        "recall",
        0
    )

    f1_score = metrics.get(
        "f1_score",
        0
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with metric2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with metric3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with metric4:

        st.metric(
            "F1 Score",
            f"{f1_score * 100:.2f}%"
        )


    if language == "Bangla":

        real_recall = metrics.get(
            "real_recall",
            0
        )

        fake_recall = metrics.get(
            "fake_recall",
            0
        )


        st.write("")


        extra1, extra2, extra3 = (
            st.columns(3)
        )


        with extra1:

            st.metric(
                "Real Recall",
                f"{real_recall * 100:.2f}%"
            )


        with extra2:

            st.metric(
                "Fake Recall",
                f"{fake_recall * 100:.2f}%"
            )


        with extra3:

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )


    st.html("""
    <div class="notice">

        The scores shown here are Machine Learning
        model outputs.

        They should not be interpreted as independent
        proof that a real-world news article is
        factually true or false.

    </div>
    """)


    st.write("")
    st.write("")


    button1, button2 = st.columns(2)


    with button1:

        if st.button(
            "Analyze Another Article",
            type="primary",
            use_container_width=True
        ):

            st.session_state.saved_headline = ""
            st.session_state.saved_article = ""

            st.session_state.page = (
                "analyzer"
            )

            st.rerun()


    with button2:

        if st.button(
            "Return Home",
            use_container_width=True
        ):

            st.session_state.saved_headline = ""
            st.session_state.saved_article = ""

            st.session_state.page = (
                "home"
            )

            st.rerun()


    st.html("""
    <div class="footer">
        ANALYSIS RESULT
        &nbsp;&nbsp; / &nbsp;&nbsp;
        MACHINE LEARNING + NLP
    </div>
    """)


# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "home":

    render_home()


elif st.session_state.page == "analyzer":

    render_analyzer()


else:

    render_result()