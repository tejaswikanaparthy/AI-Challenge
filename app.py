import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import os
import logging
from dotenv import load_dotenv
import io
import base64
from functools import lru_cache
from collections import Counter
import faiss
from sentence_transformers import SentenceTransformer
import atexit
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cleanup function for multiprocessing
def cleanup_multiprocessing():
    """Clean up multiprocessing resources on shutdown."""
    multiprocessing.active_children()
    logger.info("Cleaned up multiprocessing resources")

atexit.register(cleanup_multiprocessing)

# Set page configuration
st.set_page_config(
    page_title="Advanced App Review Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Import modules
try:
    from google_api import fetch_google_play_reviews, post_google_play_response
    from apple_api import fetch_apple_app_store_reviews, post_apple_app_store_response
except ImportError as e:
    logger.error(f"API module import failed: {e}")
    st.error("API modules not found. Please ensure google_api.py and apple_api.py exist.")
    fetch_google_play_reviews = lambda *args, **kwargs: {}
    fetch_apple_app_store_reviews = lambda *args, **kwargs: {}
    post_google_play_response = lambda *args, **kwargs: False
    post_apple_app_store_response = lambda *args, **kwargs: False

from sentiment_analysis import analyze_sentiment, categorize_review, detect_hallucination, detect_language, get_review_summary
try:
    from visualization import create_rating_chart, create_sentiment_pie, create_category_bar, create_wordcloud, create_language_pie, create_timeline_chart
except ImportError as e:
    logger.error(f"Visualization module import failed: {e}")
    def dummy_chart(df): return px.bar(title="Chart Unavailable")
    create_rating_chart = dummy_chart
    create_sentiment_pie = dummy_chart
    create_category_bar = dummy_chart
    create_wordcloud = lambda df: None
    create_language_pie = dummy_chart
    create_timeline_chart = dummy_chart

from response_generator import generate_ai_response, log_flagged_response

# Custom CSS
custom_css = """
<style>
body {
    background-color: #f8f9fa;
}
h1.dashboard-title {
    color: #f8c470;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
}
h2.section-title {
    color: #DAF7A6;
    font-size: 1.8rem;
    font-weight: 600;
}
.card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e3a8a;
}
.metric-value.positive {
    color: #10b981;
}
.metric-label {
    font-size: 1rem;
    color: #6b7280;
}
.filter-section {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.5rem;
}
.download-btn {
    display: inline-block;
    background-color: #1e3a8a;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    text-decoration: none;
    font-weight: 500;
}
.download-btn:hover {
    background-color: #1e40af;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="assets/styles.css">', unsafe_allow_html=True)

# Initialize FAISS index and SentenceTransformer
@st.cache_resource
def initialize_faiss_index():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    return model, index

embedding_model, faiss_index = initialize_faiss_index()

# Initialize session state
def initialize_session_state():
    defaults = {
        'tab': 'dashboard',
        'reviews_data': None,
        'google_creds_set': False,
        'apple_creds_set': False,
        'google_fetch_more': False,
        'apple_fetch_more': False,
        'google_next_start_index': 0,
        'apple_next_cursor': None,
        'responses': {},
        'generating_reviews': set(),
        'apple_filters': {
            'date_range': (),
            'min_rating': 1,
            'max_rating': 5,
            'sentiments': [],
            'categories': ['All'],
            'languages': ['All'],
            'applied': False
        },
        'google_filters': {
            'date_range': (),
            'min_rating': 1,
            'max_rating': 5,
            'sentiments': [],
            'categories': ['All'],
            'languages': ['All'],
            'applied': False
        },
        'apple_page': 1,
        'google_page': 1,
        'reviews_per_page': 10,
        'review_embeddings': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load credentials
def load_credentials_from_env():
    google_package_name = os.getenv('GOOGLE_PACKAGE_NAME')
    google_service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_PATH')
    if google_package_name and google_service_account_path and os.path.exists(google_service_account_path):
        st.session_state.google_package_name = google_package_name
        st.session_state.google_service_account_path = google_service_account_path
        st.session_state.google_creds_set = True
        logger.info("Google Play credentials loaded")

    apple_issuer_id = os.getenv('APPLE_ISSUER_ID')
    apple_key_id = os.getenv('APPLE_KEY_ID')
    apple_private_key_path = os.getenv('APPLE_PRIVATE_KEY_PATH')
    apple_app_id = os.getenv('APPLE_APP_ID')
    if all([apple_issuer_id, apple_key_id, apple_private_key_path, apple_app_id]) and os.path.exists(apple_private_key_path):
        st.session_state.apple_issuer_id = apple_issuer_id
        st.session_state.apple_key_id = apple_key_id
        st.session_state.apple_private_key_path = apple_private_key_path
        st.session_state.apple_app_id = apple_app_id
        with open(apple_private_key_path, 'r') as f:
            st.session_state.apple_private_key = f.read()
        st.session_state.apple_creds_set = True
        logger.info("Apple App Store credentials loaded")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        logger.info("OpenAI API key loaded")
    else:
        logger.warning("OpenAI API key not found")

    faq_csv_path = os.getenv('FAQ_CSV_PATH')
    if faq_csv_path and os.path.exists(faq_csv_path):
        st.session_state.faq_csv_path = faq_csv_path
        logger.info("FAQ CSV loaded")
    else:
        st.session_state.faq_csv_path = None
        logger.warning("FAQ CSV not found")

    brand_voice = os.getenv('BRAND_VOICE', 'friendly')
    st.session_state.brand_voice = brand_voice

# Utility functions
def format_date(date):
    if not date or pd.isna(date):
        return ""
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date, utc=True, errors='coerce')
        if pd.isna(date):
            return ""
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - date
        if delta.days == 0:
            if delta.seconds < 60:
                return "Just now"
            elif delta.seconds < 3600:
                return f"{delta.seconds // 60}m ago"
            else:
                return f"{delta.seconds // 3600}h ago"
        elif delta.days == 1:
            return "Yesterday"
        elif delta.days < 7:
            return f"{delta.days}d ago"
        elif delta.days < 30:
            return f"{delta.days // 7}w ago"
        elif delta.days < 365:
            return f"{delta.days // 30}mo ago"
        else:
            return date.strftime("%b %d, %Y")
    except Exception as e:
        logger.error(f"Date formatting error: {e}")
        return ""

@st.cache_data
def generate_csv_download(df, filename):
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        return f'<a href="data:text/csv;base64,{b64}" download="{filename}" class="download-btn">Download {filename}</a>'
    except Exception as e:
        logger.error(f"CSV download generation error: {e}")
        return ""

# FAISS functions
def update_faiss_index(df):
    if df.empty or 'text' not in df.columns:
        logger.warning("No valid data to update FAISS index")
        return
    try:
        texts = df['text'].dropna().astype(str).tolist()
        if not texts:
            logger.warning("No texts available for FAISS index")
            return
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        st.session_state.review_embeddings = embeddings
        faiss_index.reset()
        faiss_index.add(embeddings)
        logger.info(f"Updated FAISS index with {len(texts)} reviews")
    except Exception as e:
        logger.error(f"FAISS index update error: {e}")

def get_similar_reviews(review_text, k=5):
    if st.session_state.reviews_data is None or faiss_index.ntotal == 0:
        logger.warning("No reviews data or FAISS index for similar reviews")
        return []
    try:
        query_embedding = embedding_model.encode([review_text], show_progress_bar=False)
        distances, indices = faiss_index.search(query_embedding, k)
        valid_indices = [i for i in indices[0] if i < len(st.session_state.reviews_data)]
        similar_reviews_df = st.session_state.reviews_data.iloc[valid_indices]
        similar_reviews = similar_reviews_df.to_dict('records')
        logger.info(f"Found {len(similar_reviews)} similar reviews for: {review_text[:100]}...")
        return similar_reviews
    except Exception as e:
        logger.error(f"Similar reviews error: {e}")
        return []

# Main app
def main():
    load_credentials_from_env()

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #f8c470;'>App Analytics</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #6b7280;'>Navigation</h4>", unsafe_allow_html=True)
        if st.button("📊 Dashboard", key="nav_dashboard"):
            st.session_state.tab = 'dashboard'
            st.rerun()
        if st.button("📝 Reviews", key="nav_reviews"):
            st.session_state.tab = 'reviews'
            st.rerun()
        if st.button("⚙️ Settings", key="nav_settings"):
            st.session_state.tab = 'settings'
            st.rerun()
        if st.button("🚩 Flagged Responses", key="nav_flagged"):
            st.session_state.tab = 'flagged'
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #6b7280;'>API Status</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"Google: {'✅' if st.session_state.google_creds_set else '❌'}")
        with col2:
            st.markdown(f"Apple: {'✅' if st.session_state.apple_creds_set else '❌'}")

        if st.button("🔄 Refresh Data"):
            with st.spinner("Fetching reviews..."):
                fetch_all_reviews()

    if st.session_state.tab == 'dashboard':
        render_dashboard()
    elif st.session_state.tab == 'reviews':
        render_review_management()
    elif st.session_state.tab == 'settings':
        render_settings()
    elif st.session_state.tab == 'flagged':
        render_flagged_responses()

# Dashboard
def render_dashboard():
    st.markdown('<h1 class="dashboard-title">📊 App Review Analytics</h1>', unsafe_allow_html=True)

    if not st.session_state.google_creds_set and not st.session_state.apple_creds_set:
        st.error("No API credentials found. Please configure in .env file.")
        return

    if st.session_state.reviews_data is None:
        with st.spinner("Fetching reviews..."):
            fetch_all_reviews()

    if st.session_state.reviews_data is not None and not st.session_state.reviews_data.empty:
        df = st.session_state.reviews_data
        tabs = st.tabs(["Combined", "Apple App Store", "Google Play Store"])

        with tabs[0]:
            st.markdown('<h2 class="section-title">Combined Analytics</h2>', unsafe_allow_html=True)
            render_metrics(df, "Combined")
            render_charts(df, "Combined")

        with tabs[1]:
            apple_df = df[df['store'] == 'Apple App Store']
            st.markdown('<h2 class="section-title">Apple App Store</h2>', unsafe_allow_html=True)
            if st.session_state.apple_creds_set and not apple_df.empty:
                render_metrics(apple_df, "Apple App Store")
                render_charts(apple_df, "Apple App Store")
            else:
                st.info("No Apple App Store data available.")

        with tabs[2]:
            google_df = df[df['store'] == 'Google Play Store']
            st.markdown('<h2 class="section-title">Google Play Store</h2>', unsafe_allow_html=True)
            if st.session_state.google_creds_set and not google_df.empty:
                render_metrics(google_df, "Google Play Store")
                render_charts(google_df, "Google Play Store")
            else:
                st.info("No Google Play Store data available.")

    else:
        st.info("No reviews available. Using mock data for demonstration.")
        st.session_state.reviews_data = generate_mock_reviews()
        df = st.session_state.reviews_data
        render_metrics(df, "Mock Data")
        render_charts(df, "Mock Data")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #6b7280;'>Author: Tejaswi Kanaparthi</p>", unsafe_allow_html=True)

def generate_mock_reviews():
    mock_reviews = [
        {'review_id': '1', 'user_id': 'user1', 'text': 'App crashes often, very frustrating.', 'rating': 2, 'date': '2025-04-01T10:00:00Z', 'store': 'Google Play Store'},
        {'review_id': '2', 'user_id': 'user2', 'text': 'Great UI but slow performance.', 'rating': 3, 'date': '2025-04-02T12:00:00Z', 'store': 'Apple App Store'},
        {'review_id': '3', 'user_id': 'user3', 'text': 'Love the new features!', 'rating': 5, 'date': '2025-04-03T14:00:00Z', 'store': 'Google Play Store'},
        {'review_id': '4', 'user_id': 'user4', 'text': 'Login issues persist.', 'rating': 1, 'date': '2025-04-04T16:00:00Z', 'store': 'Apple App Store'},
        {'review_id': '5', 'user_id': 'user5', 'text': 'Smooth experience, highly recommend.', 'rating': 4, 'date': '2025-04-05T18:00:00Z', 'store': 'Google Play Store'},
    ]
    df = pd.DataFrame(mock_reviews)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['sentiment_analysis'] = df['text'].apply(analyze_sentiment)
    df['sentiment'] = df['sentiment_analysis'].apply(lambda x: x['sentiment'])
    df['polarity'] = df['sentiment_analysis'].apply(lambda x: x['polarity'])
    df['emotions'] = df['sentiment_analysis'].apply(lambda x: x['emotions'])
    df['category_analysis'] = df.apply(lambda x: categorize_review(x['text'], x['sentiment']), axis=1)
    df['category'] = df['category_analysis'].apply(lambda x: x['category'])
    df['category_confidence'] = df['category_analysis'].apply(lambda x: x['confidence'])
    df['language'] = df['text'].apply(detect_language)
    df['response_status'] = 'pending'
    df['hallucination_detected'] = False
    df['hallucination_feedback'] = False
    update_faiss_index(df)
    return df

def render_metrics(df, store_name):
    if df.empty:
        st.warning(f"No data available for {store_name} metrics.")
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_rating = df['rating'].mean() if 'rating' in df else 0
        st.markdown(f'<div class="metric-value">{avg_rating:.1f}</div><div class="metric-label">Avg Rating</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        review_count = len(df)
        st.markdown(f'<div class="metric-value">{review_count}</div><div class="metric-label">Total Reviews</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        positive_pct = df['sentiment'].value_counts().get('positive', 0) / review_count * 100 if review_count else 0
        st.markdown(f'<div class="metric-value positive">{positive_pct:.1f}%</div><div class="metric-label">Positive</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        hallucination_rate = df['hallucination_detected'].mean() * 100 if 'hallucination_detected' in df and not df.empty else 0
        st.markdown(f'<div class="metric-value">{hallucination_rate:.1f}%</div><div class="metric-label">Hallucination Rate</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        similar_reviews_count = 0
        if not df.empty and 'text' in df:
            similar_reviews = get_similar_reviews(df['text'].iloc[0], k=10)
            similar_reviews_count = len(similar_reviews)
        st.markdown(f'<div class="metric-value">{similar_reviews_count}</div><div class="metric-label">Similar Reviews</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_charts(df, store_name):
    if df.empty or 'text' not in df:
        st.info(f"No data for {store_name} charts.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Ratings")
        try:
            fig = create_rating_chart(df)
            st.plotly_chart(fig, use_container_width=True, key=f"{store_name}_rating_chart")
        except Exception as e:
            logger.error(f"Rating chart error: {e}")
            st.warning("Unable to display rating chart.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Sentiment")
        try:
            fig = create_sentiment_pie(df)
            st.plotly_chart(fig, use_container_width=True, key=f"{store_name}_sentiment_chart")
        except Exception as e:
            logger.error(f"Sentiment chart error: {e}")
            st.warning("Unable to display sentiment chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Categories")
        try:
            fig = create_category_bar(df)
            st.plotly_chart(fig, use_container_width=True, key=f"{store_name}_category_chart")
        except Exception as e:
            logger.error(f"Category chart error: {e}")
            st.warning("Unable to display category chart.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Word Cloud")
        try:
            fig = create_wordcloud(df)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("No word cloud generated.")
        except Exception as e:
            logger.error(f"Word cloud error: {e}")
            st.warning("Unable to display word cloud.")
        st.markdown('</div>', unsafe_allow_html=True)

    col5 = st.columns(1)[0]
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        all_emotions = []
        if 'emotions' in df:
            for emotions_dict in df['emotions']:
                if isinstance(emotions_dict, dict):
                    all_emotions.extend([(emotion, score) for emotion, score in emotions_dict.items()])
        top_emotion = max(Counter([e[0] for e in all_emotions]).items(), key=lambda x: x[1])[0] if all_emotions else 'N/A'
        st.markdown(f'<div class="metric-value">{top_emotion}</div><div class="metric-label">Dominant Emotion</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    summary = get_review_summary(df)
    logger.info(f"Summary for {store_name}: {summary}")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Topics")
        try:
            if summary.get('topics'):
                topics_df = pd.DataFrame(summary['topics'])
                if not topics_df.empty and 'topic_id' in topics_df and 'keywords' in topics_df:
                    # Ensure keywords are valid strings
                    topics_df['keywords'] = topics_df['keywords'].apply(
                        lambda x: x if isinstance(x, list) and x else ['no', 'keywords']
                    )
                    topics_df['keywords_str'] = topics_df['keywords'].apply(lambda x: ', '.join(map(str, x)))
                    topics_df['topic_id'] = topics_df['topic_id'].astype(str)  # Ensure topic_id is string for Plotly
                    # Log for debugging
                    logger.info(f"Topics DataFrame for {store_name}: {topics_df.to_dict()}")
                    fig = px.bar(
                        topics_df,
                        x='topic_id',
                        y='keywords_str',
                        title="Topics",
                        labels={'topic_id': 'Topic ID', 'keywords_str': 'Keywords'},
                        text='keywords_str',  # Display keywords on bars
                        height=400
                    )
                    fig.update_traces(textposition='auto')
                    fig.update_layout(showlegend=False, xaxis_title="Topic", yaxis_title=None)
                else:
                    logger.warning(f"Topics data missing required fields for {store_name}")
                    st.warning("Topics data missing required fields.")
                    fig = px.bar(title="Topics (Invalid Data)")
            else:
                logger.warning(f"No topics available for {store_name}")
                st.warning("No topics available.")
                fig = px.bar(title="Topics (No Data)")
            st.plotly_chart(fig, use_container_width=True, key=f"{store_name}_topics_chart")
        except Exception as e:
            logger.error(f"Topics chart error for {store_name}: {e}")
            st.warning("Unable to display topics chart.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{store_name} - Sentiment Trend")
        try:
            trend_data = summary.get('sentiment_trend', [])
            if not trend_data:
                st.info("No sentiment trend data available.")
                fig = px.line(title="Sentiment Over Time (No Data)")
            else:
                trend_df = pd.DataFrame(trend_data)
                if 'date' in trend_df:
                    trend_df['date'] = pd.to_datetime(trend_df['date'], errors='coerce', utc=True)
                    numeric_cols = trend_df.select_dtypes(include=[np.number]).columns
                    if numeric_cols.empty:
                        st.warning("No numeric data for sentiment trend.")
                        fig = px.line(title="Sentiment Over Time (Invalid Data)")
                    else:
                        fig = px.line(
                            trend_df,
                            x='date',
                            y=numeric_cols[0],
                            title="Sentiment Over Time",
                            labels={numeric_cols[0]: "Sentiment Metric"}
                        )
                else:
                    st.warning("No date data for sentiment trend.")
                    fig = px.line(title="Sentiment Over Time (Invalid Data)")
            st.plotly_chart(fig, use_container_width=True, key=f"{store_name}_trend_chart")
        except Exception as e:
            logger.error(f"Sentiment trend error: {e}")
            st.warning("Unable to display sentiment trend.")
        st.markdown('</div>', unsafe_allow_html=True)

# Review Management
def render_review_management():
    st.markdown('<h1 class="dashboard-title">📝 Review Management</h1>', unsafe_allow_html=True)

    if not st.session_state.google_creds_set and not st.session_state.apple_creds_set:
        st.error("No API credentials found.")
        return

    if st.session_state.reviews_data is None:
        with st.spinner("Fetching reviews..."):
            fetch_all_reviews()

    if st.session_state.reviews_data is not None and not st.session_state.reviews_data.empty:
        tabs = st.tabs(["Apple App Store", "Google Play Store"])
        
        with tabs[0]:
            apple_df = st.session_state.reviews_data[st.session_state.reviews_data['store'] == 'Apple App Store']
            render_store_reviews(apple_df, "Apple App Store", st.session_state.apple_filters)

        with tabs[1]:
            google_df = st.session_state.reviews_data[st.session_state.reviews_data['store'] == 'Google Play Store']
            render_store_reviews(google_df, "Google Play Store", st.session_state.google_filters)

    else:
        st.info("No reviews available. Using mock data.")
        st.session_state.reviews_data = generate_mock_reviews()
        apple_df = st.session_state.reviews_data[st.session_state.reviews_data['store'] == 'Apple App Store']
        google_df = st.session_state.reviews_data[st.session_state.reviews_data['store'] == 'Google Play Store']
        tabs = st.tabs(["Apple App Store", "Google Play Store"])
        with tabs[0]:
            render_store_reviews(apple_df, "Apple App Store", st.session_state.apple_filters)
        with tabs[1]:
            render_store_reviews(google_df, "Google Play Store", st.session_state.google_filters)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #6b7280;'>Author: Tejaswi Kanaparthi</p>", unsafe_allow_html=True)

def render_store_reviews(df, store_name, filters):
    if df.empty:
        st.warning(f"No reviews available for {store_name}.")
        return

    st.markdown(f'<h2 class="section-title">{store_name}</h2>', unsafe_allow_html=True)
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    
    with st.expander("Filters", expanded=not filters['applied']):
        col1, col2 = st.columns(2)
        with col1:
            date_key = f"{store_name}_date_range"
            date_value = filters['date_range'] if filters['date_range'] else None
            date_range = st.date_input(
                "Date Range",
                value=date_value,
                min_value=(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=365)).date(),
                max_value=datetime.datetime.now(datetime.timezone.utc).date(),
                key=date_key
            )
        with col2:
            rating_range = st.slider(
                "Rating Range",
                min_value=1,
                max_value=5,
                value=(filters['min_rating'], filters['max_rating']),
                key=f"{store_name}_rating_range"
            )
            min_rating, max_rating = rating_range

        col3, col4 = st.columns(2)
        with col3:
            sentiments = st.multiselect(
                "Sentiment",
                options=['positive', 'negative', 'neutral'],
                default=filters['sentiments'],
                key=f"{store_name}_sentiments"
            )
        with col4:
            unique_categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            categories = st.multiselect(
                "Categories",
                options=unique_categories,
                default=filters['categories'],
                key=f"{store_name}_categories"
            )

        col5, _ = st.columns([1, 1])
        with col5:
            unique_languages = ['All'] + sorted(df['language'].dropna().unique().tolist())
            languages = st.multiselect(
                "Languages",
                options=unique_languages,
                default=filters['languages'],
                key=f"{store_name}_languages"
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Apply Filters", key=f"{store_name}_apply_filters"):
                filters.update({
                    'date_range': date_range if isinstance(date_range, tuple) else (date_range, date_range) if date_range else (),
                    'min_rating': min_rating,
                    'max_rating': max_rating,
                    'sentiments': sentiments,
                    'categories': categories,
                    'languages': languages,
                    'applied': True
                })
                st.session_state[f"{store_name.lower().replace(' ', '_')}_page"] = 1
                st.rerun()
        with col_btn2:
            if st.button("Reset Filters", key=f"{store_name}_reset_filters"):
                filters.update({
                    'date_range': (),
                    'min_rating': 1,
                    'max_rating': 5,
                    'sentiments': [],
                    'categories': ['All'],
                    'languages': ['All'],
                    'applied': False
                })
                st.session_state[f"{store_name.lower().replace(' ', '_')}_page"] = 1
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df.copy()
    if filters['applied']:
        if filters['date_range']:
            start_date, end_date = filters['date_range']
            start_date = pd.Timestamp(start_date, tz='UTC')
            end_date = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
            ]
        filtered_df = filtered_df[
            (filtered_df['rating'] >= filters['min_rating']) &
            (filtered_df['rating'] <= filters['max_rating'])
        ]
        if filters['sentiments']:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(filters['sentiments'])]
        if filters['categories'] and 'All' not in filters['categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        if filters['languages'] and 'All' not in filters['languages']:
            filtered_df = filtered_df[filtered_df['language'].isin(filters['languages'])]

    if filtered_df.empty:
        st.warning("No reviews match the selected filters.")
        return

    page_key = f"{store_name.lower().replace(' ', '_')}_page"
    total_pages = max(1, (len(filtered_df) + st.session_state.reviews_per_page - 1) // st.session_state.reviews_per_page)
    current_page = max(1, min(st.session_state.get(page_key, 1), total_pages))

    start_idx = (current_page - 1) * st.session_state.reviews_per_page
    end_idx = start_idx + st.session_state.reviews_per_page
    page_df = filtered_df.iloc[start_idx:end_idx].copy()

#    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Reviews")

    if f"{store_name}_response_errors" not in st.session_state:
        st.session_state[f"{store_name}_response_errors"] = {}

    if st.button("Bulk Generate Responses", key=f"{store_name}_bulk_generate"):
        with st.spinner("Generating responses for all reviews on this page..."):
            for _, row in page_df.iterrows():
                review_id = row['review_id']
                response_key = f"response_{store_name}_{review_id}"
                if row.get('response_status', 'pending') == 'pending' and review_id not in st.session_state.generating_reviews:
                    st.session_state.generating_reviews.add(review_id)
                    try:
                        sentiment = row.get('sentiment', 'neutral') or 'neutral'
                        category = row.get('category', 'general') or 'general'
                        rating = row['rating']
                        similar_reviews = get_similar_reviews(row['text'], k=3)
                        brand_voice = st.session_state.get('brand_voice', 'friendly')
                        openai_api_key = st.session_state.get('openai_api_key', '')
                        faq_csv_path = st.session_state.get('faq_csv_path', '')
                        logger.info(f"Bulk generating for {review_id}: text='{row['text'][:50]}...', sentiment={sentiment}, category={category}, rating={rating}, similar_reviews={len(similar_reviews)}, brand_voice={brand_voice}, api_key_set={bool(openai_api_key)}, faq={faq_csv_path}")
                        if not row['text']:
                            raise ValueError("Review text is empty")
                        if not openai_api_key:
                            raise ValueError("OpenAI API key not configured")
                        response = generate_ai_response(
                            row['text'],
                            sentiment,
                            category,
                            rating,
                            similar_reviews,
                            brand_voice,
                            openai_api_key,
                            faq_csv_path
                        )
                        logger.info(f"Bulk response for {review_id}: {response[:50]}...")
                        st.session_state.responses[response_key] = response or ""
                    except Exception as e:
                        logger.error(f"Bulk response failed for {review_id}: {e}")
                        st.session_state[f"{store_name}_response_errors"][review_id] = str(e)
                    finally:
                        st.session_state.generating_reviews.remove(review_id)
            st.rerun()

    for _, row in page_df.iterrows():
        rating_display = f"{'⭐' * int(row['rating'])} ({row['rating']}/5)"
        with st.expander(f"Review by {row.get('user_id', 'Anonymous')} ({format_date(row['date'])}) {rating_display}"):
            st.markdown(f"**Rating**: {'⭐' * int(row['rating'])} ({row['rating']}/5)")
            st.markdown(f"**Sentiment**: {row.get('sentiment', 'N/A').capitalize()}")
            st.markdown(f"**Category**: {row.get('category', 'N/A')}")
            st.markdown(f"**Language**: {row.get('language', 'Unknown')}")
            st.markdown(f"**Review**: {row['text']}")
            emotions = row.get('emotions', {})
            if isinstance(emotions, dict) and emotions:
                top_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                st.markdown(f"**Top Emotion**: {top_emotion.capitalize()}")
            else:
                st.markdown("**Top Emotion**: N/A")

            similar_reviews = get_similar_reviews(row['text'], k=3)
            if similar_reviews:
                st.markdown("**Similar Reviews**:")
                for sim_review in similar_reviews:
                    st.markdown(f"- {sim_review['text']} (Rating: {sim_review['rating']}, {format_date(sim_review['date'])})")

            response_status = row.get('response_status', 'pending')
            st.markdown(f"**Response Status**: {response_status.capitalize()}")

            review_id = row['review_id']
            response_key = f"response_{store_name}_{review_id}"
            if response_key not in st.session_state.responses:
                st.session_state.responses[response_key] = ""

            if review_id in st.session_state[f"{store_name}_response_errors"]:
                st.error(f"Error generating response: {st.session_state[f'{store_name}_response_errors'][review_id]}")
                if st.button("Clear Error", key=f"clear_error_{store_name}_{review_id}"):
                    del st.session_state[f"{store_name}_response_errors"][review_id]
                    st.rerun()

            if response_status == 'pending' and review_id not in st.session_state.generating_reviews:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate AI Response", key=f"gen_{store_name}_{review_id}"):
                        st.session_state.generating_reviews.add(review_id)
                        with st.spinner("Generating response..."):
                            try:
                                sentiment = row.get('sentiment', 'neutral') or 'neutral'
                                category = row.get('category', 'general') or 'general'
                                rating = row['rating']
                                similar_reviews = get_similar_reviews(row['text'], k=3)
                                brand_voice = st.session_state.get('brand_voice', 'friendly')
                                openai_api_key = st.session_state.get('openai_api_key', '')
                                faq_csv_path = st.session_state.get('faq_csv_path', '')
                                logger.info(f"Generating for {review_id}: text='{row['text'][:50]}...', sentiment={sentiment}, category={category}, rating={rating}, similar_reviews={len(similar_reviews)}, brand_voice={brand_voice}, api_key_set={bool(openai_api_key)}, faq={faq_csv_path}")
                                if not row['text']:
                                    raise ValueError("Review text is empty")
                                if not openai_api_key:
                                    raise ValueError("OpenAI API key not configured")
                                response = generate_ai_response(
                                    row['text'],
                                    sentiment,
                                    category,
                                    rating,
                                    similar_reviews,
                                    brand_voice,
                                    openai_api_key,
                                    faq_csv_path
                                )
                                if not response:
                                    raise ValueError("Generated response is empty")
                                logger.info(f"Generated response for {review_id}: {response[:50]}...")
                                st.session_state.responses[response_key] = response
                                st.success("Response generated successfully")
                                if review_id in st.session_state[f"{store_name}_response_errors"]:
                                    del st.session_state[f"{store_name}_response_errors"][review_id]
                            except Exception as e:
                                logger.error(f"Failed to generate AI response for {review_id}: {e}")
                                st.session_state[f"{store_name}_response_errors"][review_id] = str(e)
                            finally:
                                st.session_state.generating_reviews.remove(review_id)
                            st.rerun()
                with col2:
                    if st.button("Mark as Responded", key=f"mark_{store_name}_{review_id}"):
                        filtered_df.loc[filtered_df['review_id'] == review_id, 'response_status'] = 'responded'
                        st.session_state.reviews_data.loc[
                            st.session_state.reviews_data['review_id'] == review_id, 'response_status'
                        ] = 'responded'
                        st.rerun()

            response_text = st.text_area(
                "AI-Generated Response" if response_status == 'pending' else "Response",
                value=st.session_state.responses.get(response_key, ""),
                key=f"response_text_{store_name}_{review_id}",
                disabled=response_status != 'pending'
            )

            if response_status == 'pending':
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Save Response", key=f"save_{store_name}_{review_id}"):
                        st.session_state.responses[response_key] = response_text
                        filtered_df.loc[filtered_df['review_id'] == review_id, 'response_status'] = 'saved'
                        st.session_state.reviews_data.loc[
                            st.session_state.reviews_data['review_id'] == review_id, 'response_status'
                        ] = 'saved'
                        st.rerun()
                with col2:
                    if st.button("Post Response", key=f"post_{store_name}_{review_id}"):
                        if not response_text.strip():
                            st.error("Response cannot be empty.")
                        else:
                            success = False
                            if store_name == "Google Play Store":
                                success = post_google_play_response(review_id, response_text)
                            elif store_name == "Apple App Store":
                                success = post_apple_app_store_response(review_id, response_text)
                            if success:
                                filtered_df.loc[filtered_df['review_id'] == review_id, 'response_status'] = 'posted'
                                st.session_state.reviews_data.loc[
                                    st.session_state.reviews_data['review_id'] == review_id, 'response_status'
                                ] = 'posted'
                                st.session_state.responses[response_key] = response_text
                                st.success("Response posted successfully.")
                            else:
                                st.error("Failed to post response.")
                            st.rerun()
                with col3:
                    if st.button("Flag Response", key=f"flag_{store_name}_{review_id}"):
                        if response_text.strip():
                            log_flagged_response(review_id, response_text, "Manually flagged")
                            filtered_df.loc[filtered_df['review_id'] == review_id, 'hallucination_detected'] = True
                            st.session_state.reviews_data.loc[
                                st.session_state.reviews_data['review_id'] == review_id, 'hallucination_detected'
                            ] = True
                            st.warning("Response flagged for review.")
                            st.rerun()

    st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if current_page > 1:
            if st.button("Previous", key=f"{store_name}_prev"):
                st.session_state[page_key] = current_page - 1
                st.rerun()
    with col2:
        st.markdown(f"Page {current_page} of {total_pages}")
    with col3:
        if current_page < total_pages:
            if st.button("Next", key=f"{store_name}_next"):
                st.session_state[page_key] = current_page + 1
                st.rerun()

    if store_name == "Google Play Store" and st.session_state.google_fetch_more:
        if st.button("Load More Google Reviews", key="load_more_google"):
            with st.spinner("Fetching more Google reviews..."):
                fetch_all_reviews(google_only=True)
                st.rerun()
    elif store_name == "Apple App Store" and st.session_state.apple_fetch_more:
        if st.button("Load More Apple Reviews", key="load_more_apple"):
            with st.spinner("Fetching more Apple reviews..."):
                fetch_all_reviews(apple_only=True)
                st.rerun()

    csv_link = generate_csv_download(filtered_df, f"{store_name}_reviews.csv")
    st.markdown(csv_link, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_settings():
    st.markdown('<h1 class="dashboard-title">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.subheader("Google Play Store")
    st.markdown(f"Status: {'✅ Connected' if st.session_state.google_creds_set else '❌ Not Connected'}")
    
    st.subheader("Apple App Store")
    st.markdown(f"Status: {'✅ Connected' if st.session_state.apple_creds_set else '❌ Not Connected'}")
    
    st.subheader("OpenAI API")
    st.markdown(f"Status: {'✅ Configured' if 'openai_api_key' in st.session_state else '⚠️ Not Configured'}")
    
    st.subheader("FAQ Data")
    st.markdown(f"Status: {'✅ Configured' if st.session_state.get('faq_csv_path') else '⚠️ Not Configured'}")
    
    st.subheader("FAISS Index")
    st.markdown(f"Status: {'✅ Active' if faiss_index.ntotal > 0 else '⚠️ Empty'}")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #6b7280;'>Author: Tejaswi Kanaparthi</p>", unsafe_allow_html=True)

def render_flagged_responses():
    st.markdown('<h1 class="dashboard-title">🚩 Flagged Responses</h1>', unsafe_allow_html=True)
    FLAGGED_RESPONSES_PATH = "flagged_responses.csv"
    try:
        if os.path.exists(FLAGGED_RESPONSES_PATH):
            flagged_df = pd.read_csv(FLAGGED_RESPONSES_PATH)
            st.dataframe(flagged_df)
            fig = px.histogram(flagged_df, x='reason', title="Flagged Response Reasons")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged responses yet.")
    except Exception as e:
        logger.error(f"Flagged responses error: {e}")
        st.warning("Unable to display flagged responses.")

def fetch_all_reviews():
    reviews = []
    if st.session_state.google_creds_set:
        try:
            result = fetch_google_play_reviews(
                st.session_state.google_package_name,
                st.session_state.google_service_account_path,
                max_results=100
            )
            if result and 'reviews' in result:
                for review in result['reviews']:
                    review['store'] = 'Google Play Store'
                reviews.extend(result['reviews'])
                st.session_state.google_next_start_index = result.get('next_start_index', 0)
                st.session_state.google_fetch_more = result.get('has_more', False)
                logger.info(f"Google fetch: {len(result['reviews'])} reviews")
        except Exception as e:
            logger.error(f"Google reviews fetch error: {e}")

    if st.session_state.apple_creds_set:
        try:
            result = fetch_apple_app_store_reviews(
                st.session_state.apple_issuer_id,
                st.session_state.apple_key_id,
                st.session_state.apple_private_key,
                st.session_state.apple_app_id,
                limit=100
            )
            if result and 'reviews' in result:
                for review in result['reviews']:
                    review['store'] = 'Apple App Store'
                reviews.extend(result['reviews'])
                st.session_state.apple_next_cursor = result.get('next_cursor')
                st.session_state.apple_fetch_more = result.get('has_more', False)
                logger.info(f"Apple fetch: {len(result['reviews'])} reviews")
        except Exception as e:
            logger.error(f"Apple reviews fetch error: {e}")

    if not reviews:
        logger.warning("No reviews fetched, using mock data")
        return generate_mock_reviews()

    df = pd.DataFrame(reviews)
    if df.empty:
        return generate_mock_reviews()

    df = df.drop_duplicates(subset=['review_id'], keep='last')
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['sentiment_analysis'] = df['text'].apply(analyze_sentiment)
    df['sentiment'] = df['sentiment_analysis'].apply(lambda x: x['sentiment'])
    df['polarity'] = df['sentiment_analysis'].apply(lambda x: x['polarity'])
    df['emotions'] = df['sentiment_analysis'].apply(lambda x: x['emotions'])
    df['category_analysis'] = df.apply(lambda x: categorize_review(x['text'], x['sentiment']), axis=1)
    df['category'] = df['category_analysis'].apply(lambda x: x['category'])
    df['category_confidence'] = df['category_analysis'].apply(lambda x: x['confidence'])
    df['language'] = df['text'].apply(detect_language)
    df['response_status'] = df.get('response_status', 'pending')
    df['hallucination_detected'] = df.get('hallucination_detected', False)
    df['hallucination_feedback'] = df.get('hallucination_feedback', False)
    df['response'] = df.get('response', None)
    df['user_id'] = df.get('user_id', 'unknown')
    st.session_state.reviews_data = df
    update_faiss_index(df)
    return df

def fetch_more_reviews(store_name):
    new_reviews = []
    if store_name == "Google Play Store" and st.session_state.google_creds_set:
        try:
            result = fetch_google_play_reviews(
                st.session_state.google_package_name,
                st.session_state.google_service_account_path,
                start_index=st.session_state.google_next_start_index,
                max_results=100
            )
            if result and 'reviews' in result:
                for review in result['reviews']:
                    review['store'] = 'Google Play Store'
                new_reviews.extend(result['reviews'])
                st.session_state.google_next_start_index = result.get('next_start_index', 0)
                st.session_state.google_fetch_more = result.get('has_more', False)
                logger.info(f"Google fetch_more: {len(new_reviews)} reviews")
        except Exception as e:
            logger.error(f"Google fetch_more error: {e}")

    elif store_name == "Apple App Store" and st.session_state.apple_creds_set:
        try:
            result = fetch_apple_app_store_reviews(
                st.session_state.apple_issuer_id,
                st.session_state.apple_key_id,
                st.session_state.apple_private_key,
                st.session_state.apple_app_id,
                limit=100,
                next_cursor=st.session_state.apple_next_cursor
            )
            if result and 'reviews' in result:
                for review in result['reviews']:
                    review['store'] = 'Apple App Store'
                new_reviews.extend(result['reviews'])
                st.session_state.apple_next_cursor = result.get('next_cursor')
                st.session_state.apple_fetch_more = result.get('has_more', False)
                logger.info(f"Apple fetch_more: {len(new_reviews)} reviews")
        except Exception as e:
            logger.error(f"Apple fetch_more error: {e}")

    if new_reviews:
        new_df = pd.DataFrame(new_reviews)
        new_df['date'] = pd.to_datetime(new_df['date'], utc=True, errors='coerce')
        new_df['sentiment_analysis'] = new_df['text'].apply(analyze_sentiment)
        new_df['sentiment'] = new_df['sentiment_analysis'].apply(lambda x: x['sentiment'])
        new_df['polarity'] = new_df['sentiment_analysis'].apply(lambda x: x['polarity'])
        new_df['emotions'] = new_df['sentiment_analysis'].apply(lambda x: x['emotions'])
        new_df['category_analysis'] = new_df.apply(lambda x: categorize_review(x['text'], x['sentiment']), axis=1)
        new_df['category'] = new_df['category_analysis'].apply(lambda x: x['category'])
        new_df['category_confidence'] = new_df['category_analysis'].apply(lambda x: x['confidence'])
        new_df['language'] = new_df['text'].apply(detect_language)
        new_df['response_status'] = 'pending'
        new_df['hallucination_detected'] = False
        new_df['hallucination_feedback'] = False
        new_df['response'] = None
        new_df['user_id'] = new_df.get('user_id', 'unknown')
        if st.session_state.reviews_data is not None:
            st.session_state.reviews_data = pd.concat([st.session_state.reviews_data, new_df], ignore_index=True)
        else:
            st.session_state.reviews_data = new_df
        st.session_state.reviews_data = st.session_state.reviews_data.drop_duplicates(subset=['review_id'], keep='last')
        update_faiss_index(st.session_state.reviews_data)
        st.session_state[f"{store_name.lower().replace(' ', '_')}_success_message"] = f"Loaded {len(new_reviews)} more reviews!"
        st.session_state[f"{store_name.lower().replace(' ', '_')}_page"] = 1
        st.rerun()

def filter_reviews(df, filters):
    if df.empty or not filters.get('applied', False):
        return df.copy()
    filtered_df = df.copy()
    try:
        if len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            start_date = pd.to_datetime(start_date, utc=True)
            end_date = pd.to_datetime(end_date, utc=True)
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= start_date.date()) &
                (filtered_df['date'].dt.date <= end_date.date())
            ]
        if 'rating' in filtered_df:
            filtered_df = filtered_df[
                (filtered_df['rating'] >= filters['min_rating']) &
                (filtered_df['rating'] <= filters['max_rating'])
            ]
        if filters['sentiments']:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(filters['sentiments'])]
        if 'All' not in filters['categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        if 'All' not in filters['languages']:
            filtered_df = filtered_df[filtered_df['language'].isin(filters['languages'])]
        return filtered_df
    except Exception as e:
        logger.error(f"Filter reviews error: {e}")
        return df.copy()

if __name__ == '__main__':
    main()
