import pandas as pd
import openai
import os
import logging
from datetime import datetime
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_faq(faq_csv_path):
    """Load FAQ data from CSV with encoding fallback."""
    if not faq_csv_path or not os.path.exists(faq_csv_path):
        logger.warning("FAQ CSV path invalid or not found")
        return pd.DataFrame()
    try:
        faq_df = pd.read_csv(faq_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            faq_df = pd.read_csv(faq_csv_path, encoding='windows-1252')
        except Exception as e:
            logger.error(f"FAQ CSV read error: {e}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"FAQ CSV read error: {e}")
        return pd.DataFrame()
    
    required_columns = ['question', 'answer']
    if not all(col in faq_df.columns for col in required_columns):
        logger.error(f"FAQ CSV missing required columns: {required_columns}")
        return pd.DataFrame()
    
    logger.info(f"Loaded FAQ with {len(faq_df)} entries")
    return faq_df

def match_faq(review_text, faq_df, review_category=None, sentiment=None, rating=None):
    """Match review to relevant FAQs using fuzzy matching."""
    if faq_df.empty:
        return []
    
    matched_faqs = []
    review_text_lower = review_text.lower()
    
    for _, row in faq_df.iterrows():
        question = str(row['question']).lower()
        answer = str(row['answer'])
        faq_category = str(row.get('category', '')).lower()
        
        # Fuzzy matching score
        score = fuzz.partial_ratio(review_text_lower, question)
        
        # Boost for category match
        if review_category and faq_category and review_category.lower() in faq_category:
            score += 20
        # Boost for sentiment and rating
        if sentiment == 'negative' and any(kw in question for kw in ['issue', 'problem', 'fix']):
            score += 10
        if rating and rating <= 2:
            score += 10 if 'troubleshoot' in question or 'fix' in question else 0
        
        if score > 50:
            matched_faqs.append({
                'question': row['question'],
                'answer': answer,
                'score': score
            })
    
    matched_faqs = sorted(matched_faqs, key=lambda x: x['score'], reverse=True)[:2]
    logger.info(f"Matched {len(matched_faqs)} FAQs for review: {review_text[:50]}...")
    return matched_faqs

def generate_ai_response(review_text, sentiment, category, rating, similar_reviews, brand_voice, openai_api_key, faq_csv_path):
    """Generate tailored response using OpenAI with FAQ and context."""
    if not openai_api_key:
        logger.error("No OpenAI API key provided")
        return "Thank you for your review. We're here to help—please contact support for assistance."
    
    try:
        # Initialize OpenAI client with minimal arguments
        client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        if sentiment == 'positive':
            return "Thank you for your kind words! We're thrilled you enjoy the app."
        elif sentiment == 'negative':
            return f"We're sorry to hear about your experience with {category.lower()}. Please try restarting the app or contact support for help."
        else:
            return "Thanks for your feedback! We're here to assist with any questions."
    
    faq_df = load_faq(faq_csv_path)
    matched_faqs = match_faq(review_text, faq_df, category, sentiment, rating)
    
    faq_context = "\n".join(
        f"Q: {faq['question']} A: {faq['answer']}" for faq in matched_faqs
    ) if matched_faqs else "No relevant FAQs available."
    
    similar_context = "\n".join(
        f"- {r['text'][:100]}" for r in similar_reviews if 'text' in r
    ) if similar_reviews else "No similar reviews."
    
    prompt = f"""
You are a {brand_voice} customer support agent for a mobile app. Craft a concise, empathetic, and professional response to the user review below. Tailor the response to the review's specific concerns, sentiment, and rating, using relevant FAQ information when applicable. If similar issues are mentioned in other reviews, acknowledge the pattern subtly without quoting them directly. Avoid generic phrases like "we're reviewing your concerns."

**Review**: {review_text}
**Sentiment**: {sentiment}
**Category**: {category}
**Rating**: {rating} stars
**Similar Reviews**:
{similar_context}
**FAQ Reference**:
{faq_context}

**Instructions**:
- Keep the response under 100 words.
- Match the {brand_voice} tone (e.g., friendly, professional).
- For positive reviews (4-5 stars), express gratitude and encourage continued use.
- For negative reviews (1-2 stars), apologize, suggest a specific solution from FAQs if relevant, and offer further assistance.
- For neutral reviews (3 stars), acknowledge feedback and provide helpful guidance.
- Ensure the response feels personal and directly addresses the review's content.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable and empathetic app support agent."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.6,
            top_p=0.9
        )
        resp_text = response.choices[0].message.content.strip()
        logger.info(f"Generated response for '{review_text[:50]}...': {resp_text[:50]}...")
        return resp_text
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        if sentiment == 'positive':
            return "Thank you for your kind words! We're thrilled you enjoy the app."
        elif sentiment == 'negative':
            return f"We're sorry to hear about your experience with {category.lower()}. Please try restarting the app or contact support for help."
        else:
            return "Thanks for your feedback! We're here to assist with any questions."

def log_flagged_response(review_text, response, reason):
    """Log flagged responses to a CSV file."""
    flagged_data = {
        'review_text': review_text,
        'response': response,
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }
    flagged_df = pd.DataFrame([flagged_data])
    flagged_file = "flagged_responses.csv"
    
    try:
        if os.path.exists(flagged_file):
            flagged_df.to_csv(flagged_file, mode='a', header=False, index=False)
        else:
            flagged_df.to_csv(flagged_file, index=False)
        logger.info(f"Logged flagged response for review: {review_text[:50]}...")
    except Exception as e:
        logger.error(f"Failed to log flagged response: {e}")
