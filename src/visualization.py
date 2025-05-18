import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_match_radar_chart(resume_data):
    """
    Create a radar chart showing match scores across different categories.
    
    Args:
        resume_data (dict): Resume data with match scores
        
    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """
    categories = ['Overall Match', 'Skills', 'Education', 'Experience']
    
    # Get scores from resume data
    values = [
        resume_data['match_score'],
        resume_data['skills_score'],
        resume_data['education_score'],
        resume_data['experience_score']
    ]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Score',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Match Score Breakdown",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    return fig

def create_skills_comparison_chart(required_skills, resume_skills):
    """
    Create a horizontal bar chart comparing required skills to resume skills.
    
    Args:
        required_skills (list): List of required skills from job description
        resume_skills (list): List of skills from resume
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Find matching and missing skills
    matching_skills = [skill for skill in resume_skills if skill in required_skills]
    missing_skills = [skill for skill in required_skills if skill not in resume_skills]
    
    # Prepare data for chart
    skills = matching_skills + missing_skills
    status = ['Match'] * len(matching_skills) + ['Missing'] * len(missing_skills)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Skill': skills,
        'Status': status,
        'Value': [1] * len(skills)
    })
    
    # Create figure
    fig = px.bar(
        df,
        y='Skill',
        x='Value',
        color='Status',
        color_discrete_map={'Match': 'green', 'Missing': 'red'},
        title='Skills Comparison',
        labels={'Value': ''},
        orientation='h'
    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showticklabels=False)
    )
    
    return fig

def create_ranking_bar_chart(ranked_resumes):
    """
    Create a bar chart showing overall ranking of resumes.
    
    Args:
        ranked_resumes (list): List of ranked resumes with scores
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Prepare data for chart
    filenames = [resume['filename'] for resume in ranked_resumes]
    scores = [resume['match_score'] for resume in ranked_resumes]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Resume': filenames,
        'Match Score': scores
    })
    
    # Create figure
    fig = px.bar(
        df,
        x='Resume',
        y='Match Score',
        title='Resume Rankings',
        color='Match Score',
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Resume',
        yaxis_title='Match Score (%)',
        yaxis=dict(range=[0, 100])
    )
    
    return fig
