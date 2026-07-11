import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Box Office Revenue Predictor", page_icon="🎬", layout="wide")

# ---------- GLOBAL STYLE ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #16141b 0%, #0b0b10 55%, #08080b 100%);
    color: #F5EFE0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #100e14;
    border-right: 1px solid #2a2530;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    color: #D4AF37;
    font-size: 1.8rem;
}

/* Headings marquee-style */
h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 1.5px;
    color: #F5EFE0 !important;
}
h1 { color: #D4AF37 !important; }

/* Body text */
p, li, span, label {
    color: #C9C2B4;
}

/* Film-strip divider */
.film-divider {
    height: 22px;
    margin: 28px 0;
    background-image: repeating-linear-gradient(
        90deg,
        #2a2530 0px, #2a2530 14px,
        transparent 14px, transparent 28px
    );
    background-position: center;
    background-size: 28px 10px;
    background-repeat: repeat-x;
    opacity: 0.6;
}

/* Ticket-stub metric card */
.ticket {
    background: #17151C;
    border: 1px dashed #D4AF37;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    margin-bottom: 10px;
}
.ticket::before, .ticket::after {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    background: #0b0b10;
    border-radius: 50%;
    top: 50%;
    transform: translateY(-50%);
}
.ticket::before { left: -9px; }
.ticket::after { right: -9px; }
.ticket-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #9C948A;
    margin-bottom: 4px;
}
.ticket-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #D4AF37;
}

/* Hero */
.hero-tagline {
    font-family: 'Inter', sans-serif;
    color: #9C948A;
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    border: 1px solid #9B2C3B;
    color: #E88A97;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}

/* Buttons */
.stButton>button {
    background: #D4AF37;
    color: #0B0B10;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 6px;
}
.stButton>button:hover {
    background: #E8C766;
    color: #0B0B10;
}
           
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.markdown('<div class="sidebar-title">🎬 BOX OFFICE AI</div>', unsafe_allow_html=True)
st.sidebar.caption("Smart revenue forecasting for films")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset Explorer", "📈 Visualizations", "🤖 Revenue Prediction", "🏆 Model Performance", "👨‍💻 About Project"],
    label_visibility="collapsed"
)

# ---------- HOME PAGE ----------
if page == "🏠 Home":
    st.markdown('<span class="hero-badge">NOW PREDICTING</span>', unsafe_allow_html=True)
    st.title("BOX OFFICE REVENUE PREDICTOR")
    st.markdown(
        '<p class="hero-tagline">Every film starts with a budget and a bet. This tool uses machine learning '
        'trained on 4,800+ real releases to estimate how a movie\'s box office revenue might play out — '
        'before a single ticket is sold.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="ticket"><div class="ticket-label">Movies Analyzed</div><div class="ticket-value">3,913</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="ticket"><div class="ticket-label">Model Accuracy (R²)</div><div class="ticket-value">0.68</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="ticket"><div class="ticket-label">Avg. Error</div><div class="ticket-value">$59.8M</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="ticket"><div class="ticket-label">Features Used</div><div class="ticket-value">5</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)
    st.markdown("Use the sidebar to explore the dataset, view visual insights, or predict revenue for a hypothetical movie.")

elif page == "📊 Dataset Explorer":
    st.title("DATASET EXPLORER")
    st.markdown('<p class="hero-tagline">Browse the cleaned dataset used to train the model — 3,913 films after removing incomplete records.</p>', unsafe_allow_html=True)
    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    movies_clean = pd.read_csv("data/tmdb_5000_movies.csv")
    movies_clean = movies_clean[~((movies_clean['budget'] == 0) & (movies_clean['revenue'] == 0))]

    col1, col2 = st.columns(2)
    with col1:
        min_budget = st.slider("Minimum Budget ($)", 0, int(movies_clean['budget'].max()), 0, step=1000000)
    with col2:
        min_rating = st.slider("Minimum Vote Average", 0.0, 10.0, 0.0, step=0.5)

    filtered = movies_clean[(movies_clean['budget'] >= min_budget) & (movies_clean['vote_average'] >= min_rating)]

    st.markdown(f"**{len(filtered)} movies** match your filters")
    st.dataframe(
        filtered[['title', 'budget', 'revenue', 'runtime', 'vote_average', 'popularity']].sort_values('revenue', ascending=False),
        use_container_width=True,
        height=400
    )

elif page == "📈 Visualizations":
    st.title("VISUALIZATIONS")
    st.markdown('<p class="hero-tagline">Visual patterns behind what drives box office revenue.</p>', unsafe_allow_html=True)
    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    movies_clean = pd.read_csv("data/tmdb_5000_movies.csv")
    movies_clean = movies_clean[~((movies_clean['budget'] == 0) & (movies_clean['revenue'] == 0))]

    tab1, tab2 = st.tabs(["Budget vs Revenue", "Popularity vs Revenue"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0B0B10')
        ax.set_facecolor('#0B0B10')
        ax.scatter(movies_clean['budget'], movies_clean['revenue'], alpha=0.5, color='#D4AF37')
        ax.set_xlabel('Budget ($)', color='#C9C2B4')
        ax.set_ylabel('Revenue ($)', color='#C9C2B4')
        ax.tick_params(colors='#9C948A')
        for spine in ax.spines.values():
            spine.set_color('#2a2530')
        st.pyplot(fig)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0B0B10')
        ax2.set_facecolor('#0B0B10')
        ax2.scatter(movies_clean['popularity'], movies_clean['revenue'], alpha=0.5, color='#9B2C3B')
        ax2.set_xlabel('Popularity', color='#C9C2B4')
        ax2.set_ylabel('Revenue ($)', color='#C9C2B4')
        ax2.tick_params(colors='#9C948A')
        for spine in ax2.spines.values():
            spine.set_color('#2a2530')
        st.pyplot(fig2)

elif page == "🤖 Revenue Prediction":
    st.title("REVENUE PREDICTION")
    st.markdown('<p class="hero-tagline">Enter a hypothetical movie\'s details and get an instant revenue estimate.</p>', unsafe_allow_html=True)
    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    model = joblib.load("models/revenue_predictor.pkl")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget ($)", min_value=0, value=50000000, step=1000000)
        runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120)
        popularity = st.slider("Popularity Score", 0.0, 500.0, 50.0)
    with col2:
        vote_average = st.slider("Expected Vote Average", 0.0, 10.0, 6.5)
        vote_count = st.number_input("Expected Vote Count", min_value=0, value=1000, step=100)

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    if st.button("🎬 Predict Revenue"):
        input_data = pd.DataFrame([[budget, popularity, runtime, vote_average, vote_count]],
                                    columns=['budget', 'popularity', 'runtime', 'vote_average', 'vote_count'])
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)  # revenue can't be negative

        st.markdown(
            f'<div class="ticket" style="max-width:400px;"><div class="ticket-label">Predicted Revenue</div>'
            f'<div class="ticket-value" style="font-size:2.2rem;">${prediction:,.0f}</div></div>',
            unsafe_allow_html=True
        )

        roi = (prediction - budget) / budget * 100 if budget > 0 else 0
        if roi > 0:
            st.success(f"Estimated profit margin: {roi:.1f}% above budget")
        else:
            st.warning(f"Estimated loss: {roi:.1f}% below budget")

elif page == "🏆 Model Performance":
    st.title("MODEL PERFORMANCE")
    st.markdown('<p class="hero-tagline">How well does the model actually predict? Evaluated on 783 unseen test movies.</p>', unsafe_allow_html=True)
    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="ticket"><div class="ticket-label">R² Score</div><div class="ticket-value">0.6765</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="ticket"><div class="ticket-label">Mean Absolute Error</div><div class="ticket-value">$59.8M</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="ticket"><div class="ticket-label">Algorithm</div><div class="ticket-value" style="font-size:1.1rem;">Linear Regression</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)
    st.markdown("### What R² = 0.68 means")
    st.write("The model explains roughly 68% of the variation in box office revenue using just 5 features: budget, popularity, runtime, vote average, and vote count. The remaining 32% is driven by factors not captured here — marketing spend, star power, release timing, competition, and word of mouth.")

    st.markdown("### Where the model struggles")
    st.write("Predictions are more accurate for low-to-mid budget films, since most of the training data falls in that range. Blockbuster outliers (very high revenue) are harder to predict precisely, since fewer examples exist to learn from.")

elif page == "👨‍💻 About Project":
    st.title("ABOUT THE PROJECT")
    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)

    st.markdown("### The Problem")
    st.write("Studios and independent filmmakers alike need to estimate potential box office performance before greenlighting a project. This tool offers a data-driven starting point using historical patterns.")

    st.markdown("### The Data")
    st.write("Built on the TMDB 5000 Movie Dataset — 4,803 films with budget, revenue, genre, popularity, and rating data. After removing 890 records with missing budget/revenue information, 3,913 films were used to train the model.")

    st.markdown("### The Approach")
    st.write("A Linear Regression model was trained on five numeric features: budget, popularity, runtime, vote average, and vote count. The dataset was split 80/20 for training and testing, and evaluated using R² and Mean Absolute Error.")

    st.markdown("### Tech Stack")
    st.write("Python · Pandas · Scikit-learn · Matplotlib · Streamlit")

    st.markdown('<div class="film-divider"></div>', unsafe_allow_html=True)
    st.caption("Built as an academic ML project — Box Office Revenue Predictor")