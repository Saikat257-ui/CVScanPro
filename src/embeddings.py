import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Use TF-IDF vectorizer as a simple alternative to sentence transformers
vectorizer = None

def load_model():
    """
    Load the TF-IDF vectorizer as a simpler alternative to sentence transformers.
    """
    global vectorizer
    try:
        vectorizer = TfidfVectorizer(max_features=384)
    except Exception as e:
        print(f"Error initializing TF-IDF vectorizer: {str(e)}")
        raise RuntimeError("Failed to initialize TF-IDF vectorizer")

def get_embedding(text):
    """
    Generate embeddings for a given text using TF-IDF.
    
    Args:
        text (str): Input text
        
    Returns:
        numpy.ndarray: Text embedding vector
    """
    global vectorizer
    
    if vectorizer is None:
        load_model()
    
    if not text:
        # Return zero vector if text is empty
        return np.zeros(384)  # Default dimension
    
    # Generate embeddings using TF-IDF
    # Fit and transform for each text (not ideal for production, but works for demo)
    try:
        # For the first call, fit the vectorizer
        if not hasattr(vectorizer, 'vocabulary_'):
            embedding = vectorizer.fit_transform([text]).toarray()[0]
        else:
            # For subsequent calls, use transform only
            embedding = vectorizer.transform([text]).toarray()[0]
            
        # Ensure the embedding is of expected size
        if len(embedding) < 384:
            # Pad with zeros if needed
            embedding = np.pad(embedding, (0, 384 - len(embedding)))
        elif len(embedding) > 384:
            # Truncate if needed
            embedding = embedding[:384]
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return np.zeros(384)

def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1 (numpy.ndarray): First embedding vector
        embedding2 (numpy.ndarray): Second embedding vector
        
    Returns:
        float: Cosine similarity score (between 0 and 1)
    """
    # Check for zero vectors
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Ensure the result is between 0 and 1
    similarity = max(0.0, min(1.0, similarity))
    
    return similarity

def get_section_embeddings(text, sections):
    """
    Generate embeddings for specific sections of a text.
    
    Args:
        text (str): Full text
        sections (list): List of section names/keywords to look for
        
    Returns:
        dict: Dictionary of section embeddings
    """
    section_embeddings = {}
    
    # Simple section extraction based on keywords
    for section in sections:
        pattern = r'(?i)' + section + r'[:\s]+(.*?)(?=\n\n|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            section_text = ' '.join(matches)
            section_embeddings[section] = get_embedding(section_text)
        else:
            section_embeddings[section] = np.zeros(384)  # Default dimension
    
    return section_embeddings
