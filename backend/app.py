from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for demo (use database in production)
candidates_db = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

def extract_resume_info(text):
    """Extract structured information from resume text"""
    info = {
        'skills': extract_skills(text),
        'experience': extract_experience(text),
        'education': extract_education(text),
        'email': extract_email(text),
        'phone': extract_phone(text)
    }
    return info

def extract_skills(text):
    """Extract skills from resume text"""
    # Common technical skills
    skills_keywords = [
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Swift',
        'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring',
        'SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
        'Machine Learning', 'Deep Learning', 'AI', 'Data Analysis',
        'Git', 'Agile', 'Scrum', 'DevOps', 'CI/CD'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in skills_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return found_skills

def extract_experience(text):
    """Extract years of experience from resume"""
    # Look for patterns like "5 years", "5+ years", etc.
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*:\s*(\d+)\+?\s*years?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0

def extract_education(text):
    """Extract education level from resume"""
    education_keywords = {
        'phd': 'PhD',
        'ph.d': 'PhD',
        'doctorate': 'PhD',
        'master': 'Master',
        'm.s.': 'Master',
        'm.s': 'Master',
        'mba': 'Master',
        'bachelor': 'Bachelor',
        'b.s.': 'Bachelor',
        'b.s': 'Bachelor',
        'b.tech': 'Bachelor',
        'b.e.': 'Bachelor'
    }
    
    text_lower = text.lower()
    for keyword, level in education_keywords.items():
        if keyword in text_lower:
            return level
    
    return 'Unknown'

def extract_email(text):
    """Extract email from resume"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text):
    """Extract phone number from resume"""
    phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else None

def calculate_match_score(resume_text, job_description):
    """Calculate similarity score between resume and job description"""
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Resume Analyzer API is running'})

@app.route('/api/upload', methods=['POST'])
def upload_resumes():
    """Upload and process multiple resume files"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    processed_resumes = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                continue
            
            # Extract structured information
            info = extract_resume_info(text)
            
            candidate = {
                'id': len(candidates_db) + 1,
                'filename': filename,
                'text': text,
                'skills': info['skills'],
                'experience': info['experience'],
                'education': info['education'],
                'email': info['email'],
                'phone': info['phone']
            }
            
            candidates_db.append(candidate)
            processed_resumes.append({
                'filename': filename,
                'skills': info['skills'],
                'experience': info['experience'],
                'education': info['education']
            })
            
            # Clean up file after processing
            os.remove(file_path)
    
    return jsonify({
        'message': f'Successfully processed {len(processed_resumes)} resumes',
        'resumes': processed_resumes
    })

@app.route('/api/search', methods=['POST'])
def search_candidates():
    """Search candidates by keywords or job description"""
    data = request.json
    query = data.get('query', '')
    job_description = data.get('job_description', query)
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    # Calculate match scores for all candidates
    results = []
    for candidate in candidates_db:
        match_score = calculate_match_score(candidate['text'], job_description)
        
        results.append({
            'id': candidate['id'],
            'filename': candidate['filename'],
            'skills': candidate['skills'],
            'experience': candidate['experience'],
            'education': candidate['education'],
            'email': candidate['email'],
            'match_score': match_score
        })
    
    # Sort by match score
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return jsonify({
        'query': query,
        'total_results': len(results),
        'candidates': results
    })

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics for all uploaded resumes"""
    if not candidates_db:
        return jsonify({'error': 'No resumes uploaded yet'}), 404
    
    # Experience breakdown
    experience_ranges = {'0-5': 0, '5-10': 0, '10+': 0}
    for candidate in candidates_db:
        exp = candidate['experience']
        if exp <= 5:
            experience_ranges['0-5'] += 1
        elif exp <= 10:
            experience_ranges['5-10'] += 1
        else:
            experience_ranges['10+'] += 1
    
    # Education distribution
    education_dist = {}
    for candidate in candidates_db:
        edu = candidate['education']
        education_dist[edu] = education_dist.get(edu, 0) + 1
    
    # Skills analysis
    all_skills = {}
    for candidate in candidates_db:
        for skill in candidate['skills']:
            all_skills[skill] = all_skills.get(skill, 0) + 1
    
    top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'total_candidates': len(candidates_db),
        'experience_breakdown': experience_ranges,
        'education_distribution': education_dist,
        'top_skills': dict(top_skills),
        'candidates': [{
            'id': c['id'],
            'filename': c['filename'],
            'experience': c['experience'],
            'education': c['education'],
            'skills_count': len(c['skills'])
        } for c in candidates_db]
    })

@app.route('/api/candidate/<int:candidate_id>', methods=['GET'])
def get_candidate_detail(candidate_id):
    """Get detailed information for a specific candidate"""
    candidate = next((c for c in candidates_db if c['id'] == candidate_id), None)
    
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404
    
    return jsonify({
        'id': candidate['id'],
        'filename': candidate['filename'],
        'skills': candidate['skills'],
        'experience': candidate['experience'],
        'education': candidate['education'],
        'email': candidate['email'],
        'phone': candidate['phone']
    })

@app.route('/api/reset', methods=['POST'])
def reset_database():
    """Reset the in-memory database (for testing)"""
    global candidates_db
    candidates_db = []
    return jsonify({'message': 'Database reset successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)