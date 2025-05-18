import re
import spacy
from collections import defaultdict

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not available, use a smaller model or provide instructions
    nlp = spacy.blank("en")
    print("Warning: 'en_core_web_sm' model not found. Using blank model.")
    print("To download the model, run: python -m spacy download en_core_web_sm")

# Lists of education and skill-related terms
EDUCATION_TERMS = [
    'bachelor', 'master', 'phd', 'doctorate', 'degree', 'mba', 'bsc', 'msc', 'b.a', 'm.a', 
    'b.s', 'm.s', 'bachelor of', 'master of', 'diploma', 'certification', 'certificate', 'university',
    'college', 'institute', 'school', 'academy', 'education'
]

COMMON_SKILLS = [
    # Programming Languages
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'typescript', 'golang', 'scala',
    # Web Development
    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net', 'laravel',
    # Data Science & AI
    'machine learning', 'deep learning', 'nlp', 'data analysis', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 
    'numpy', 'r', 'sql', 'nosql', 'statistics', 'data visualization', 'tableau', 'power bi', 'big data', 'hadoop',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'devops', 'ci/cd',
    # Business & Management
    'project management', 'agile', 'scrum', 'kanban', 'leadership', 'team management', 'product management',
    'business analysis', 'marketing', 'sales', 'customer service', 'communication',
    # Office & Productivity
    'microsoft office', 'excel', 'word', 'powerpoint', 'g suite', 'google docs',
    # Other Technical
    'rest api', 'graphql', 'microservices', 'blockchain', 'cybersecurity', 'networking', 'linux', 'unix',
    # Common Industry-Specific
    'accounting', 'finance', 'hr', 'human resources', 'healthcare', 'marketing', 'sales', 'customer service',
    'supply chain', 'logistics', 'seo', 'content marketing', 'social media'
]

# Pattern for years of experience
EXPERIENCE_PATTERN = r'(\d+)[\+]?\s*(year|yr|years|yrs)'

def extract_entities(text):
    """
    Extract entities like skills, education, and experience from text.
    
    Args:
        text (str): Preprocessed text
        
    Returns:
        dict: Dictionary containing extracted entities
    """
    if not text:
        return {"skills": [], "education": [], "experience": []}
    
    entities = {
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience": extract_experience(text)
    }
    
    return entities

def extract_skills(text):
    """
    Extract skills from text using NER and skill keyword matching.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of extracted skills
    """
    skills = set()
    
    # Use spaCy for NER
    doc = nlp(text)
    
    # Extract skills based on NER tags
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG", "GPE") and len(ent.text) > 1:
            skills.add(ent.text.lower())
    
    # Extract skills based on common skill terms
    for skill in COMMON_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            skills.add(skill.lower())
    
    # Look for programming languages or frameworks that might be missed
    code_pattern = r'\b(programming|coding|development|developer|engineer)\b'
    if re.search(code_pattern, text, re.IGNORECASE):
        for token in doc:
            if token.is_alpha and len(token.text) > 1 and token.text.lower() not in skills:
                if token.text.lower() in COMMON_SKILLS:
                    skills.add(token.text.lower())
    
    return list(skills)

def extract_education(text):
    """
    Extract education details from text.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of extracted education details
    """
    education = set()
    
    # Use regex patterns to find education-related text
    doc = nlp(text)
    sentences = [sent.text.lower() for sent in doc.sents]
    
    for sentence in sentences:
        for term in EDUCATION_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', sentence):
                # Look for education degree in the sentence
                degree_pattern = r'\b(bachelor|master|phd|doctorate|mba|bsc|msc|b\.a|m\.a|b\.s|m\.s|diploma|certificate)\b'
                degree_match = re.search(degree_pattern, sentence, re.IGNORECASE)
                
                # Look for field of study
                field_pattern = r'\b(computer science|engineering|business|management|marketing|finance|economics|accounting|psychology|biology|chemistry|physics|mathematics|communications|english|history|philosophy|political science|sociology)\b'
                field_match = re.search(field_pattern, sentence, re.IGNORECASE)
                
                if degree_match:
                    degree = degree_match.group(0)
                    if field_match:
                        education.add(f"{degree} in {field_match.group(0)}")
                    else:
                        education.add(degree)
                else:
                    # Just extract the sentence containing education term
                    clean_sentence = re.sub(r'[^\w\s]', ' ', sentence)
                    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
                    if len(clean_sentence.split()) < 10:  # Only add if it's reasonably short
                        education.add(clean_sentence)
    
    return list(education)

def extract_experience(text):
    """
    Extract professional experience details from text.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of extracted experience details
    """
    experience = set()
    
    # Extract years of experience
    years_exp = re.findall(EXPERIENCE_PATTERN, text, re.IGNORECASE)
    if years_exp:
        for year, unit in years_exp:
            experience.add(f"{year} {unit} of experience")
    
    # Extract job titles and responsibilities
    doc = nlp(text)
    
    # Look for job titles
    job_titles = [
        'manager', 'developer', 'engineer', 'analyst', 'director', 'coordinator', 'specialist',
        'assistant', 'associate', 'consultant', 'supervisor', 'administrator', 'architect',
        'designer', 'technician', 'officer', 'representative', 'lead', 'head'
    ]
    
    for token in doc:
        if token.text.lower() in job_titles and token.i > 0:
            # Look for job title with preceding token (e.g., "senior developer")
            if doc[token.i-1].is_alpha:
                job_title = f"{doc[token.i-1].text} {token.text}"
                experience.add(job_title.lower())
            else:
                experience.add(token.text.lower())
    
    # Extract company names (organizations)
    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text.split()) < 5:  # Limit to reasonably short org names
            experience.add(f"worked at {ent.text.lower()}")
    
    return list(experience)
