import os
import PyPDF2
import docx
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_resume(file_path):
    """
    Parse resume files (PDF or DOCX) and extract text.
    
    Args:
        file_path: Path to the resume file
        
    Returns:
        str: Extracted text from the resume
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return parse_pdf(file_path)
        elif file_extension == '.docx':
            return parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error parsing resume {file_path}: {str(e)}")
        raise

def parse_pdf(file_path):
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        
        # If text extraction failed, log warning
        if not text.strip():
            logger.warning(f"No text extracted from PDF {file_path}")
            return "No text could be extracted from this PDF file."
            
        return text
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {str(e)}")
        raise

def parse_docx(file_path):
    """
    Extract text from DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        str: Extracted text from the DOCX
    """
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # If text extraction failed, log warning
        if not text.strip():
            logger.warning(f"No text extracted from DOCX {file_path}")
            return "No text could be extracted from this DOCX file."
            
        return text
    except Exception as e:
        logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
        raise
