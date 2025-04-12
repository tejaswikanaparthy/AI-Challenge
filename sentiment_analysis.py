import pandas as pd
import numpy as np
import torch
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import logging
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Limit threads to avoid semaphore issues
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    emotion_analyzer = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sentiment_analyzer = None
    emotion_analyzer = None
    embedding_model = None
    vader = None

def analyze_sentiment(text):
    """Analyze sentiment and emotions with transformer primary, VADER fallback."""
    if not text or not isinstance(text, str):
        logger.warning("Invalid text for sentiment analysis")
        return {'sentiment': 'neutral', 'polarity': 0.0, 'emotions': {}}
    
    try:
        # Transformer-based sentiment
        if sentiment_analyzer:
            result = sentiment_analyzer(text, truncation=True, max_length=512)
            label = result[0]['label'].lower()
            score = result[0]['score']
            sentiment = 'positive' if label == 'positive' else 'negative' if label == 'negative' else 'neutral'
            polarity = score if sentiment == 'positive' else -score if sentiment == 'negative' else 0.0
        elif vader:
            scores = vader.polarity_scores(text)
            compound = scores['compound']
            sentiment = 'positive' if compound > 0.05 else 'negative' if compound < -0.05 else 'neutral'
            polarity = compound
        else:
            raise ValueError("No sentiment analyzer available")
        
        # Emotions
        emotions = {}
        if emotion_analyzer:
            emotion_results = emotion_analyzer(text, truncation=True, max_length=512)[0]
            emotions = {res['label']: res['score'] for res in emotion_results}
        elif vader:
            emotions = {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        
        logger.info(f"Sentiment for '{text[:30]}...': {sentiment}, polarity: {polarity}")
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'emotions': emotions
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error for '{text[:30]}...': {e}")
        return {'sentiment': 'neutral', 'polarity': 0.0, 'emotions': {}}

def categorize_review(text, sentiment):
    """Categorize review using embeddings and keywords."""
    if not text or not isinstance(text, str):
        logger.warning("Invalid text for categorization")
        return {'category': 'General', 'confidence': 0.5}
    
    try:
        categories = {
            'Technical Issues': ['crash', 'bug', 'error', 'freeze', 'not working', 'slow', 'lag'],
            'Feature Request': ['add', 'want', 'wish', 'need', 'feature', 'option', 'mode'],
            'Performance': ['speed', 'slow', 'fast', 'laggy', 'performance', 'battery'],
            'Usability': ['confusing', 'hard to use', 'interface', 'navigation', 'design'],
            'Account': ['login', 'sign in', 'password', 'account', 'profile'],
            'General': ['app', 'experience', 'overall', 'great', 'bad']
        }
        
        text_lower = text.lower()
        scores = {cat: 0 for cat in categories}
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        if sentiment == 'negative':
            scores['Technical Issues'] += 2
            scores['General'] -= 1
        elif sentiment == 'positive':
            scores['General'] += 1
        
        if embedding_model:
            category_keywords = [' '.join(kw) for kw in categories.values()]
            embeddings = embedding_model.encode([text] + category_keywords, show_progress_bar=False)
            text_emb = embeddings[0]
            category_embs = embeddings[1:]
            similarities = cosine_similarity([text_emb], category_embs)[0]
            for i, cat in enumerate(categories):
                scores[cat] += similarities[i] * 5
        
        total = sum(scores.values())
        if total == 0:
            return {'category': 'General', 'confidence': 0.5}
        
        top_category = max(scores, key=scores.get)
        confidence = scores[top_category] / total if total > 0 else 0.5
        
        logger.info(f"Categorized '{text[:30]}...' as {top_category} (confidence: {confidence:.2f})")
        return {
            'category': top_category,
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Categorization error for '{text[:30]}...': {e}")
        return {'category': 'General', 'confidence': 0.5}

def detect_hallucination(response, review_text, similar_reviews):
    """Detect hallucinations using embeddings and content checks."""
    if not response or not review_text:
        logger.warning("Invalid input for hallucination detection")
        return False
    
    try:
        if embedding_model:
            texts = [response, review_text] + [r['text'] for r in similar_reviews if 'text' in r]
            if len(texts) < 2:
                logger.warning("Insufficient texts for hallucination detection")
                return False
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            response_emb = embeddings[0]
            context_embs = embeddings[1:]
            similarities = cosine_similarity([response_emb], context_embs)[0]
            avg_similarity = np.mean(similarities)
            if avg_similarity < 0.6:
                logger.info(f"Hallucination detected: similarity {avg_similarity:.2f}")
                return True
        
        logger.info(f"No hallucination in response: {response[:50]}...")
        return False
    except Exception as e:
        logger.error(f"Hallucination detection error: {e}")
        return False

def detect_language(text):
    """Detect language using langdetect."""
    if not text or not isinstance(text, str):
        logger.warning("Invalid text for language detection")
        return 'en'
    
    try:
        lang = detect(text)
        logger.info(f"Detected language for '{text[:30]}...': {lang}")
        return lang
    except LangDetectException:
        logger.warning(f"Language detection failed for '{text[:30]}...'")
        return 'en'

def get_review_summary(df):
    """Generate topics and sentiment trend from reviews."""
    if df.empty or len(df) < 3 or 'text' not in df:
        logger.warning("Not enough valid reviews for summary")
        return {
            'topics': [{'topic_id': 0, 'keywords': ['no', 'data']}],
            'sentiment_trend': []
        }
    
    try:
        # Topic modeling
        texts = df['text'].dropna().astype(str).tolist()
        if len(texts) < 3:
            logger.warning("Too few texts for topic modeling")
            return {
                'topics': [{'topic_id': 0, 'keywords': ['insufficient', 'data']}],
                'sentiment_trend': []
            }
        
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0:  # No features after vectorization
            logger.warning("No valid features for topic modeling")
            return {
                'topics': [{'topic_id': 0, 'keywords': ['no', 'features']}],
                'sentiment_trend': []
            }
        
        n_components = min(3, len(texts))
        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        lda.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Ensure at least some keywords, even if weak
            top_indices = topic.argsort()[-5:]
            keywords = [feature_names[i] for i in top_indices if i < len(feature_names)]
            if not keywords:
                keywords = ['topic', 'empty']  # Fallback for empty keywords
            topics.append({'topic_id': topic_idx, 'keywords': keywords})
            logger.info(f"Topic {topic_idx}: {keywords}")
        
        # Sentiment trend
        trend = []
        if 'date' in df and 'polarity' in df:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df = df.dropna(subset=['date', 'polarity'])
            if not df.empty:
                trend = df.groupby(df['date'].dt.date).agg({
                    'polarity': 'mean'
                }).reset_index().rename(columns={'date': 'date', 'polarity': 'polarity'})
                trend['date'] = pd.to_datetime(trend['date'])
                logger.info(f"Generated sentiment trend with {len(trend)} points")
        
        return {
            'topics': topics,
            'sentiment_trend': trend.to_dict('records') if not trend.empty else []
        }
    except Exception as e:
        logger.error(f"Review summary error: {e}")
        return {
            'topics': [{'topic_id': 0, 'keywords': ['error', 'occurred']}],
            'sentiment_trend': []
        }
