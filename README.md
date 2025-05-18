# AI Resume Shortlisting System 🧠

An intelligent system that leverages Natural Language Processing (NLP) to automatically analyze and rank resumes against job descriptions, helping recruiters streamline their candidate selection process.

## 🌟 Features

- **Job Description Analysis**: Parse and extract key requirements from job descriptions
- **Resume Processing**: Support for both PDF and DOCX resume formats
- **Intelligent Matching**: 
  - Skills matching
  - Education qualification analysis
  - Experience level evaluation
- **Interactive Results**:
  - Ranked list of candidates
  - Match percentage scores
  - Visual representations of matches
  - Detailed candidate analysis
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## 🛠️ Technologies Used

- Python 3.11+
- Streamlit
- NLTK
- spaCy
- PyPDF2
- python-docx
- Pandas
- Plotly
- scikit-learn

## 📋 Prerequisites

- Python 3.11 or higher
- pip package manager

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Follow the steps in the application:
   - Paste the job description
   - Upload candidate resumes (PDF/DOCX)
   - Click "Process and Rank Resumes"
   - View the results in the Results tab

## 📊 Features in Detail

### Job Description Analysis
- Extracts key requirements, skills, and qualifications
- Processes natural language to understand job requirements

### Resume Processing
- Supports multiple file formats (PDF, DOCX)
- Extracts text content while maintaining structure
- Identifies key sections and information

### Matching Algorithm
- Skills matching with semantic understanding
- Education level comparison
- Experience duration analysis
- Weighted scoring system

### Results Visualization
- Overall match percentage
- Individual category scores
- Interactive charts and graphs
- Detailed candidate profiles

## 🔧 Configuration

The application can be configured using the `.streamlit/config.toml` file:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── resume_parser.py   # Resume parsing logic
│   ├── text_processor.py  # Text processing utilities
│   ├── entity_extractor.py# Entity extraction logic
│   ├── embeddings.py      # Text embedding generation
│   ├── ranker.py         # Resume ranking algorithm
│   └── visualization.py   # Data visualization components
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- Built with Streamlit Framework
- Uses spaCy for Natural Language Processing
- Leverages scikit-learn for machine learning capabilities
