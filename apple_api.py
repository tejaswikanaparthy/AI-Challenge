import requests
import jwt
import time
from datetime import datetime, timedelta
import json
import logging
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_apple_jwt(issuer_id, key_id, private_key_content):
    """
    Generate a JSON Web Token (JWT) for Apple App Store Connect API
    
    Args:
        issuer_id (str): The issuer ID from App Store Connect
        key_id (str): The key ID from App Store Connect
        private_key_content (str): The content of the private key (.p8 file)
        
    Returns:
        str: JWT token
    """
    try:
        # Current time and expiration time (20 minutes from now)
        current_time = int(time.time())
        expiration_time = current_time + 1200  # 20 minutes
        
        # Clean up the private key content if needed
        if "-----BEGIN PRIVATE KEY-----" not in private_key_content:
            private_key_content = f"-----BEGIN PRIVATE KEY-----\n{private_key_content}\n-----END PRIVATE KEY-----"
        
        # Load the private key
        key = load_pem_private_key(
            private_key_content.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        
        # Prepare the JWT payload
        payload = {
            'iss': issuer_id,
            'exp': expiration_time,
            'aud': 'appstoreconnect-v1',
            'iat': current_time
        }
        
        # Generate the JWT token
        token = jwt.encode(
            payload,
            key,
            algorithm='ES256',
            headers={
                'kid': key_id,
                'typ': 'JWT'
            }
        )
        
        # Handle different jwt library versions (some return bytes, some return string)
        if isinstance(token, bytes):
            return token.decode('utf-8')
        return token
    
    except Exception as e:
        logger.error(f"Error generating Apple JWT: {str(e)}")
        raise

def fetch_apple_app_store_reviews(issuer_id, key_id, private_key_content, app_id, limit=50, next_cursor=None):
    """
    Fetch reviews from Apple App Store using the App Store Connect API
    
    Args:
        issuer_id (str): The issuer ID from App Store Connect
        key_id (str): The key ID from App Store Connect
        private_key_content (str): The content of the private key (.p8 file)
        app_id (str): The App ID from App Store Connect
        limit (int): Maximum number of reviews to fetch per page
        next_cursor (str): Cursor for pagination
        
    Returns:
        dict: Dictionary with reviews and pagination info
    """
    try:
        # Generate JWT token for authentication
        token = generate_apple_jwt(issuer_id, key_id, private_key_content)
        
        # Apple App Store Connect API URL for customer reviews
        url = f"https://api.appstoreconnect.apple.com/v1/apps/{app_id}/customerReviews"
        
        # Request parameters
        params = {
            'limit': limit,
            'sort': '-createdDate'  # Sort by creation date in descending order
        }
        
        # Add cursor for pagination if provided
        if next_cursor:
            params['after'] = next_cursor
        
        # Request headers
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # Make the API request
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            logger.error(f"Apple App Store API error: {response.status_code} - {response.text}")
            return {'reviews': [], 'next_cursor': None, 'has_more': False}
        
        # Parse the response
        response_data = response.json()
        reviews_data = response_data.get('data', [])
        
        # Get pagination info
        links = response_data.get('links', {})
        next_page_url = links.get('next', None)
        has_more = next_page_url is not None
        
        # Extract next cursor from next page URL if it exists
        next_cursor = None
        if has_more and next_page_url:
            import urllib.parse
            parsed_url = urllib.parse.urlparse(next_page_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            next_cursor = query_params.get('after', [None])[0]
        
        # Process each review
        processed_reviews = []
        
        for review in reviews_data:
            review_attrs = review.get('attributes', {})
            review_relationships = review.get('relationships', {})
            
            # Extract review date
            review_date_str = review_attrs.get('createdDate')
            review_date = datetime.fromisoformat(review_date_str.replace('Z', '+00:00')) if review_date_str else datetime.now()
            
            # Get developer response if exists
            response_status = 'pending'
            developer_response = None
            
            if 'developerResponse' in review_relationships:
                dev_response_data = review_relationships['developerResponse'].get('data')
                if dev_response_data:
                    response_status = 'responded'
                    
                    # If included responses are in the data, try to get the full response text
                    if 'included' in response_data:
                        for included_item in response_data['included']:
                            if (included_item['type'] == 'customerReviewResponses' and
                                included_item['id'] == dev_response_data['id']):
                                developer_response = included_item.get('attributes', {}).get('responseBody')
                                break
            
            # Process the review
            processed_review = {
                'review_id': review.get('id', ''),
                'date': review_date,
                'user_id': review_attrs.get('reviewerNickname', 'Anonymous'),
                'store': 'Apple App Store',
                'language': review_attrs.get('territory', 'US'),  # Use territory as language approximation
                'text': review_attrs.get('body', ''),
                'title': review_attrs.get('title', ''),
                'rating': review_attrs.get('rating', 0),
                'response_status': response_status,
                'developer_response': developer_response,
                'app_version': review_attrs.get('appVersionString', 'Unknown')
            }
            
            processed_reviews.append(processed_review)
        
        return {
            'reviews': processed_reviews,
            'next_cursor': next_cursor,
            'has_more': has_more
        }
        
    except Exception as e:
        logger.error(f"Error fetching Apple App Store reviews: {str(e)}")
        return {'reviews': [], 'next_cursor': None, 'has_more': False}

def post_apple_app_store_response(issuer_id, key_id, private_key_content, review_id, response_text):
    """
    Post a response to a review on the Apple App Store
    
    Args:
        issuer_id (str): The issuer ID from App Store Connect
        key_id (str): The key ID from App Store Connect
        private_key_content (str): The content of the private key (.p8 file)
        review_id (str): The ID of the review to respond to
        response_text (str): The response text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate JWT token for authentication
        token = generate_apple_jwt(issuer_id, key_id, private_key_content)
        
        # Apple App Store Connect API URL for customer review responses
        url = "https://api.appstoreconnect.apple.com/v1/customerReviewResponses"
        
        # Request headers
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # Request payload
        payload = {
            "data": {
                "type": "customerReviewResponses",
                "attributes": {
                    "responseBody": response_text
                },
                "relationships": {
                    "review": {
                        "data": {
                            "type": "customerReviews",
                            "id": review_id
                        }
                    }
                }
            }
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code in [201, 200]:
            logger.info(f"Successfully replied to Apple review {review_id}")
            return True
        else:
            logger.error(f"Error posting Apple App Store response: {response.status_code} - {response.text}")
            return False
        
    except Exception as e:
        logger.error(f"Error posting Apple App Store response: {str(e)}")
        return False
