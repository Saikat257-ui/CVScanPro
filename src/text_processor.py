import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Initialize RegexpTokenizer as fallback
WORD_TOKENIZER = RegexpTokenizer(r'\w+')

# Download necessary NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Warning: Error downloading NLTK resources: {str(e)}")

# Initialize stopwords and lemmatizer
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation, 
    removing stop words, and lemmatizing.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    
    # Remove special characters and digits (keeping spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize with fallback
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(f"Warning: Using fallback tokenizer due to error: {str(e)}")
        tokens = WORD_TOKENIZER.tokenize(text)
    
    # Remove stopwords and lemmatize
    filtered_tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS and len(token) > 1]
    
    # Join tokens back into a single string
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text

def extract_keywords(text, n=1):
    """
    Extract keywords from text (unigrams, bigrams, etc.)
    
    Args:
        text (str): Input text
        n (int): n-gram size (1 for unigrams, 2 for bigrams, etc.)
        
    Returns:
        list: List of extracted keywords
    """
    if not text:
        return []
    
    tokens = word_tokenize(text)
    keywords = []
    
    for i in range(len(tokens) - n + 1):
        keyword = ' '.join(tokens[i:i+n])
        keywords.append(keyword)
    
    return keywords
