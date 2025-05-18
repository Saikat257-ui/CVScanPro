import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
from src.resume_parser import parse_resume
from src.text_processor import preprocess_text
from src.entity_extractor import extract_entities
from src.embeddings import get_embedding
from src.ranker import rank_resumes
from src.visualization import create_match_radar_chart

# Set page configuration
st.set_page_config(
    page_title="AI Resume Shortlisting System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables if not already present
if 'parsed_resumes' not in st.session_state:
    st.session_state.parsed_resumes = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'ranked_resumes' not in st.session_state:
    st.session_state.ranked_resumes = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def process_job_description(job_description):
    """Process the job description text."""
    if not job_description.strip():
        st.error("Please enter a job description.")
        return None
    
    # Preprocess text
    processed_jd = preprocess_text(job_description)
    
    # Extract entities
    entities_jd = extract_entities(processed_jd)
    
    # Get embedding
    embedding_jd = get_embedding(processed_jd)
    
    return {
        "text": job_description,
        "processed_text": processed_jd,
        "entities": entities_jd,
        "embedding": embedding_jd
    }

def process_resumes(uploaded_files):
    """Process the uploaded resume files."""
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        return []
    
    parsed_resumes = []
    with st.spinner("Processing resumes... This may take a minute."):
        for uploaded_file in uploaded_files:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            try:
                # Parse the resume
                resume_text = parse_resume(file_path)
                
                # Preprocess text
                processed_text = preprocess_text(resume_text)
                
                # Extract entities
                entities = extract_entities(processed_text)
                
                # Get embedding
                embedding = get_embedding(processed_text)
                
                parsed_resumes.append({
                    "filename": uploaded_file.name,
                    "text": resume_text,
                    "processed_text": processed_text,
                    "entities": entities,
                    "embedding": embedding
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Clean up temporary file
            os.unlink(file_path)
    
    return parsed_resumes

def main():
    # Header
    st.title("🧠 AI-powered Resume Shortlisting System")
    st.markdown("""
    This system uses Natural Language Processing to match resumes to job descriptions, 
    helping recruiters identify the most suitable candidates.
    """)
    
    # Create two tabs
    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Results"])
    
    with tab1:
        st.header("Step 1: Enter Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            value=st.session_state.job_description,
            height=200,
            help="Paste the complete job description including requirements, responsibilities, etc."
        )
        
        st.header("Step 2: Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX resumes (multiple files allowed):",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload candidate resumes in either PDF or DOCX format."
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            process_button = st.button("Process and Rank Resumes", type="primary")
        
        if process_button:
            # Process job description
            processed_jd = process_job_description(job_description)
            if processed_jd is None:
                return
            
            # Process resumes
            parsed_resumes = process_resumes(uploaded_files)
            if not parsed_resumes:
                return
                
            # Store in session state
            st.session_state.job_description = job_description
            st.session_state.parsed_resumes = parsed_resumes
            
            # Rank resumes against the job description
            with st.spinner("Ranking resumes..."):
                ranked_resumes = rank_resumes(processed_jd, parsed_resumes)
                st.session_state.ranked_resumes = ranked_resumes
                st.session_state.show_results = True
            
            # Switch to results tab
            st.rerun()
    
    with tab2:
        if not st.session_state.show_results:
            st.info("Please upload and process resumes to see results here.")
        else:
            st.header("Resume Ranking Results")
            st.markdown("Here are the resumes ranked by their match to the job description:")
            
            # Display results as a table
            results_df = pd.DataFrame([
                {
                    "Resume": resume["filename"],
                    "Match Score": f"{resume['match_score']:.2f}%", 
                    "Skills Match": f"{resume['skills_score']:.2f}%",
                    "Education Match": f"{resume['education_score']:.2f}%",
                    "Experience Match": f"{resume['experience_score']:.2f}%"
                } 
                for resume in st.session_state.ranked_resumes
            ])
            
            st.dataframe(
                results_df,
                column_config={
                    "Resume": st.column_config.TextColumn("Resume"),
                    "Match Score": st.column_config.TextColumn("Match Score"),
                    "Skills Match": st.column_config.TextColumn("Skills Match"),
                    "Education Match": st.column_config.TextColumn("Education Match"),
                    "Experience Match": st.column_config.TextColumn("Experience Match")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Create a bar chart for overall match scores
            fig = px.bar(
                results_df, 
                x="Resume", 
                y=[resume["match_score"] for resume in st.session_state.ranked_resumes],
                labels={"y": "Match Score (%)", "x": "Resume"},
                title="Overall Match Scores",
                color=[resume["match_score"] for resume in st.session_state.ranked_resumes],
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed resume view
            st.header("Detailed Resume Analysis")
            selected_resume = st.selectbox(
                "Select resume for detailed analysis:",
                options=[resume["filename"] for resume in st.session_state.ranked_resumes]
            )
            
            # Find the selected resume
            selected_resume_data = next(
                (r for r in st.session_state.ranked_resumes if r["filename"] == selected_resume), 
                None
            )
            
            if selected_resume_data:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Match Breakdown")
                    radar_chart = create_match_radar_chart(selected_resume_data)
                    st.plotly_chart(radar_chart, use_container_width=True)
                    
                    st.subheader("Key Matching Skills")
                    for skill in selected_resume_data["matching_skills"][:10]:  # Show top 10 skills
                        st.markdown(f"✅ {skill}")
                    
                    st.subheader("Missing Skills")
                    for skill in selected_resume_data["missing_skills"][:5]:  # Show top 5 missing skills
                        st.markdown(f"❌ {skill}")
                
                with col2:
                    st.subheader("Resume Content")
                    with st.expander("Show Full Resume Text", expanded=False):
                        st.text_area(
                            "Resume Content",
                            value=selected_resume_data["text"],
                            height=300,
                            disabled=True
                        )
                    
                    st.subheader("Extracted Information")
                    st.markdown("##### Skills")
                    skills_text = ", ".join(selected_resume_data["entities"]["skills"])
                    st.markdown(f"*{skills_text}*")
                    
                    st.markdown("##### Education")
                    education_text = ", ".join(selected_resume_data["entities"]["education"])
                    st.markdown(f"*{education_text}*")
                    
                    st.markdown("##### Experience")
                    experience_text = ", ".join(selected_resume_data["entities"]["experience"][:5])  # Limit to 5 experiences
                    st.markdown(f"*{experience_text}*")

if __name__ == "__main__":
    main()
