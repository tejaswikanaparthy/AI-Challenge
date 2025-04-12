import requests
import json
import logging
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_google_service(service_account_json):
    """
    Create an authorized Google API service using service account credentials
    
    Args:
        service_account_json (str): Path to service account JSON file or JSON string
        
    Returns:
        service: Authorized Google API service
    """
    try:
        # Check if input is a file path or a JSON string
        if isinstance(service_account_json, str) and service_account_json.endswith('.json'):
            credentials = service_account.Credentials.from_service_account_file(
                service_account_json,
                scopes=['https://www.googleapis.com/auth/androidpublisher']
            )
        else:
            # If it's a JSON string
            service_account_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/androidpublisher']
            )
        
        # Build the service
        service = build('androidpublisher', 'v3', credentials=credentials)
        return service
    
    except Exception as e:
        logger.error(f"Error getting Google API service: {str(e)}")
        raise

def fetch_google_play_reviews(package_name, service_account_json, max_results=50, start_index=0):
    """
    Fetch reviews from Google Play Store using the Google Play Developer API
    
    Args:
        package_name (str): The package name of the app (e.g., com.example.app)
        service_account_json (str): Path to service account JSON file or JSON string
        max_results (int): Maximum number of reviews to fetch
        start_index (int): Index to start fetching reviews from
        
    Returns:
        list: List of processed review dictionaries
    """
    try:
        # Get the Google service
        service = get_google_service(service_account_json)
        
        # Create a request to fetch reviews
        reviews_resource = service.reviews()
        request = reviews_resource.list(
            packageName=package_name,
            maxResults=max_results,
            startIndex=start_index,
            translationLanguage='en'
        )
        
       
        response = request.execute()
        
        
        reviews_data = response.get('reviews', [])
        processed_reviews = []
        
        for review in reviews_data:
            # Extract user comment information
            user_comment = review.get('comments', [{}])[0].get('userComment', {})
            
            # Extract developer reply if exists
            developer_comment = None
            if len(review.get('comments', [])) > 1:
                developer_comment = review.get('comments', [{}])[1].get('developerComment', {})
            
            # Convert timestamp to datetime
            timestamp = int(user_comment.get('lastModified', {}).get('seconds', 0))
            review_date = datetime.fromtimestamp(timestamp)
            
            # Determine response status
            response_status = 'responded' if developer_comment else 'pending'
            
            # Extract review text and handle translations if available
            review_text = user_comment.get('text', '')
            if user_comment.get('originalText'):
                # Use original text if available
                review_text = user_comment.get('originalText', '')
            
            # Process the review
            processed_review = {
                'review_id': review.get('reviewId', ''),
                'date': review_date,
                'user_id': review.get('authorName', 'Anonymous'),
                'store': 'Google Play Store',
                'language': user_comment.get('reviewerLanguage', 'en'),
                'text': review_text,
                'rating': user_comment.get('starRating', 0),
                'response_status': response_status,
                'app_version': user_comment.get('appVersionName', 'Unknown'),
                'device': user_comment.get('deviceMetadata', {}).get('productName', 'Unknown'),
                'thumbs_up_count': user_comment.get('thumbsUpCount', 0),
                'thumbs_down_count': user_comment.get('thumbsDownCount', 0)
            }
            
            processed_reviews.append(processed_review)
        
        # Check if there are more reviews to fetch
        has_more_reviews = 'tokenPagination' in response
        next_start_index = start_index + len(reviews_data) if has_more_reviews else None
        
        return {
            'reviews': processed_reviews,
            'next_start_index': next_start_index,
            'has_more': has_more_reviews
        }
    
    except Exception as e:
        logger.error(f"Error fetching Google Play reviews: {str(e)}")
        return {'reviews': [], 'next_start_index': None, 'has_more': False}

def post_google_play_response(package_name, service_account_json, review_id, response_text):
    """
    Post a response to a review on Google Play Store
    
    Args:
        package_name (str): The package name of the app
        service_account_json (str): Path to service account JSON file or JSON string
        review_id (str): The ID of the review to respond to
        response_text (str): The response text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the Google service
        service = get_google_service(service_account_json)
        
        # Create a request to reply to a review
        reviews_resource = service.reviews()
        request = reviews_resource.reply(
            packageName=package_name,
            reviewId=review_id,
            body={'replyText': response_text}
        )
        
        # Execute the request
        response = request.execute()
        
        # Check if the response contains the reply
        if 'result' in response and 'replyText' in response['result']:
            logger.info(f"Successfully replied to review {review_id}")
            return True
        else:
            logger.warning(f"Response didn't contain expected fields: {response}")
            return False
    
    except Exception as e:
        logger.error(f"Error posting Google Play response: {str(e)}")
        return False
