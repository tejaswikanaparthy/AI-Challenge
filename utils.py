import pandas as pd
import os
from datetime import datetime
import csv
import logging
import json

def load_faq_data(file_path=None):
    """
    Load FAQ data from a CSV file
    
    Args:
        file_path (str): Path to the CSV file containing FAQ data
        
    Returns:
        dict: Dictionary of FAQ data
    """
    try:
        # If no file path specified, return empty dict
        if not file_path or not os.path.exists(file_path):
            return {}
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns
            if 'key' in df.columns and 'question' in df.columns and 'answer' in df.columns:
                # Convert to dictionary
                faq_data = {}
                for _, row in df.iterrows():
                    faq_data[row['key']] = {
                        'question': row['question'],
                        'answer': row['answer']
                    }
                return faq_data
            else:
                logging.error("FAQ CSV file missing required columns (key, question, answer)")
                return {}
                
        elif ext.lower() == '.json':
            # Load JSON file
            with open(file_path, 'r') as f:
                return json.load(f)
        
        else:
            logging.error(f"Unsupported file format: {ext}")
            return {}
            
    except Exception as e:
        logging.error(f"Error loading FAQ data: {str(e)}")
        return {}

def save_faq_data(faq_data, file_path):
    """
    Save FAQ data to a CSV file
    
    Args:
        faq_data (dict): Dictionary of FAQ data
        file_path (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check file extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            # Convert to dataframe
            data = []
            for key, faq in faq_data.items():
                data.append({
                    'key': key,
                    'question': faq['question'],
                    'answer': faq['answer']
                })
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
        elif ext.lower() == '.json':
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(faq_data, f, indent=4)
        
        else:
            logging.error(f"Unsupported file format: {ext}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error saving FAQ data: {str(e)}")
        return False

def format_date(date):
    """
    Format datetime object to readable string
    
    Args:
        date (datetime): Datetime object
        
    Returns:
        str: Formatted date string
    """
    try:
        if not date:
            return ""
            
        # Convert to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # Calculate relative time
        now = datetime.now()
        delta = now - date
        
        if delta.days == 0:
            # Today
            if delta.seconds < 60:
                return "Just now"
            elif delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.days == 1:
            # Yesterday
            return "Yesterday"
        elif delta.days < 7:
            # This week
            return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
        elif delta.days < 30:
            # This month
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif delta.days < 365:
            # This year
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            # More than a year
            return date.strftime("%b %d, %Y")
            
    except Exception as e:
        logging.error(f"Error formatting date: {str(e)}")
        return str(date)

def detect_keyword_matches(text, keywords):
    """
    Detect keyword matches in text
    
    Args:
        text (str): Text to search in
        keywords (list): List of keywords to search for
        
    Returns:
        list: List of matched keywords
    """
    if not text or not keywords:
        return []
        
    text_lower = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in text_lower]

def clean_text(text):
    """
    Clean text by removing special characters and extra whitespace
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Remove special characters
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_app_version(version_str):
    """
    Extract standardized app version from version string
    
    Args:
        version_str (str): Version string (e.g., "1.2.3", "v1.2.3", "Version 1.2.3")
        
    Returns:
        str: Standardized version string (e.g., "1.2.3")
    """
    if not version_str:
        return ""
        
    # Extract version using regex
    import re
    match = re.search(r'(\d+\.\d+(?:\.\d+)?)', str(version_str))
    
    if match:
        return match.group(1)
    
    return str(version_str)

def create_directory_if_not_exists(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        logging.error(f"Error creating directory: {str(e)}")
        return False
