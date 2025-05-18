import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import calculate_similarity

def rank_resumes(job_description, parsed_resumes):
    """
    Rank resumes based on their similarity to the job description.
    
    Args:
        job_description (dict): Processed job description with embeddings
        parsed_resumes (list): List of processed resumes with embeddings
        
    Returns:
        list: Ranked list of resumes with scores
    """
    if not job_description or not parsed_resumes:
        return []
    
    ranked_resumes = []
    
    for resume in parsed_resumes:
        # Calculate overall similarity score using embeddings
        similarity_score = calculate_similarity(
            job_description["embedding"], 
            resume["embedding"]
        )
        
        # Calculate skills match
        skills_score = calculate_skills_match(
            job_description["entities"]["skills"],
            resume["entities"]["skills"]
        )
        
        # Calculate education match
        education_score = calculate_education_match(
            job_description["entities"]["education"],
            resume["entities"]["education"]
        )
        
        # Calculate experience match
        experience_score = calculate_experience_match(
            job_description["entities"]["experience"],
            resume["entities"]["experience"]
        )
        
        # Combine scores (weighted average)
        overall_score = (
            similarity_score * 0.4 + 
            skills_score * 0.3 + 
            education_score * 0.15 + 
            experience_score * 0.15
        ) * 100  # Convert to percentage
        
        # Find matching and missing skills
        matching_skills = [
            skill for skill in resume["entities"]["skills"] 
            if any(job_skill.lower() in skill.lower() or skill.lower() in job_skill.lower() 
                  for job_skill in job_description["entities"]["skills"])
        ]
        
        missing_skills = [
            skill for skill in job_description["entities"]["skills"]
            if not any(skill.lower() in resume_skill.lower() or resume_skill.lower() in skill.lower()
                      for resume_skill in resume["entities"]["skills"])
        ]
        
        # Add to ranked list
        ranked_resumes.append({
            "filename": resume["filename"],
            "text": resume["text"],
            "processed_text": resume["processed_text"],
            "entities": resume["entities"],
            "embedding": resume["embedding"],
            "match_score": overall_score,
            "skills_score": skills_score * 100,
            "education_score": education_score * 100,
            "experience_score": experience_score * 100,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills
        })
    
    # Sort resumes by match score (descending)
    ranked_resumes.sort(key=lambda x: x["match_score"], reverse=True)
    
    return ranked_resumes

def calculate_skills_match(job_skills, resume_skills):
    """
    Calculate skill match score between job and resume.
    
    Args:
        job_skills (list): Skills from job description
        resume_skills (list): Skills from resume
        
    Returns:
        float: Skill match score (between 0 and 1)
    """
    if not job_skills:
        return 0.0
    
    if not resume_skills:
        return 0.0
    
    # Convert to lowercase for case-insensitive matching
    job_skills_lower = [skill.lower() for skill in job_skills]
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    
    # Count matching skills (partial matches allowed)
    matching_skills = 0
    for job_skill in job_skills_lower:
        if any(job_skill in resume_skill or resume_skill in job_skill for resume_skill in resume_skills_lower):
            matching_skills += 1
    
    # Calculate score as proportion of job skills matched
    score = matching_skills / len(job_skills)
    
    return score

def calculate_education_match(job_education, resume_education):
    """
    Calculate education match score between job and resume.
    
    Args:
        job_education (list): Education requirements from job description
        resume_education (list): Education details from resume
        
    Returns:
        float: Education match score (between 0 and 1)
    """
    if not job_education:
        return 1.0  # If job doesn't specify education, assume full match
    
    if not resume_education:
        return 0.0
    
    # Convert to lowercase for case-insensitive matching
    job_edu_lower = [edu.lower() for edu in job_education]
    resume_edu_lower = [edu.lower() for edu in resume_education]
    
    # Look for keyword matches across education terms
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'certificate']
    matches = 0
    
    for job_edu in job_edu_lower:
        for resume_edu in resume_edu_lower:
            # Check for direct or partial matches
            if job_edu in resume_edu or resume_edu in job_edu:
                matches += 1
                break
            
            # Check for keyword matches
            for keyword in education_keywords:
                if keyword in job_edu and keyword in resume_edu:
                    matches += 0.5
                    break
    
    # Calculate score
    score = min(1.0, matches / len(job_education))
    
    return score

def calculate_experience_match(job_experience, resume_experience):
    """
    Calculate experience match score between job and resume.
    
    Args:
        job_experience (list): Experience requirements from job description
        resume_experience (list): Experience details from resume
        
    Returns:
        float: Experience match score (between 0 and 1)
    """
    if not job_experience:
        return 1.0  # If job doesn't specify experience, assume full match
    
    if not resume_experience:
        return 0.0
    
    # Convert to lowercase for case-insensitive matching
    job_exp_lower = [exp.lower() for exp in job_experience]
    resume_exp_lower = [exp.lower() for exp in resume_experience]
    
    # Look for years of experience matches
    import re
    
    years_pattern = r'(\d+)[\+]?\s*(year|yr|years|yrs)'
    
    job_years = []
    for exp in job_exp_lower:
        matches = re.findall(years_pattern, exp)
        if matches:
            for match in matches:
                job_years.append(int(match[0]))
    
    resume_years = []
    for exp in resume_exp_lower:
        matches = re.findall(years_pattern, exp)
        if matches:
            for match in matches:
                resume_years.append(int(match[0]))
    
    # If years are specified in both, compare them
    if job_years and resume_years:
        max_job_years = max(job_years)
        max_resume_years = max(resume_years)
        
        if max_resume_years >= max_job_years:
            return 1.0
        else:
            return max_resume_years / max_job_years
    
    # Look for keyword matches
    matches = 0
    for job_exp in job_exp_lower:
        for resume_exp in resume_exp_lower:
            # Check for partial matches
            if job_exp in resume_exp or resume_exp in job_exp:
                matches += 1
                break
    
    # Calculate score
    score = min(1.0, matches / len(job_experience))
    
    return score
