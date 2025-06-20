import streamlit as st
import re
import io
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMMON_SKILLS = {
    'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift', 'typescript', 'kotlin', 'golang', 'rust'],
    'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express', 'bootstrap', 'jquery'],
    'databases': ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis', 'elasticsearch', 'dynamodb', 'firebase'],
    'devops': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'git', 'ci/cd', 'github actions'],
    'data_science': ['machine learning', 'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'r', 'tableau', 'power bi'],
    'soft_skills': ['teamwork', 'leadership', 'communication', 'problem solving', 'critical thinking', 'time management']
}

ALL_SKILLS = [skill for category in COMMON_SKILLS.values() for skill in category]

@st.cache_resource
def load_nlp_model():
    try:
        import spacy
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            st.warning("Downloading spaCy model. This may take a moment...")
            import subprocess
            subprocess.call([
                "python", "-m", "spacy", "download", "en_core_web_sm"
            ])
            return spacy.load('en_core_web_sm')
    except ImportError:
        st.error("spaCy is not installed. Installing required packages...")
        import subprocess
        subprocess.call(["pip", "install", "spacy"])
        subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        return spacy.load('en_core_web_sm')

@st.cache_resource
def load_pdf_module():
    try:
        import PyPDF2
        return PyPDF2
    except ImportError:
        st.warning("Installing PDF processing library...")
        import subprocess
        subprocess.call(["pip", "install", "PyPDF2"])
        import PyPDF2
        return PyPDF2

def extract_text_from_pdf(pdf_file):
    PyPDF2 = load_pdf_module()
    
    text = ""
    try:
        pdf_bytes = io.BytesIO(pdf_file.read())
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\-]', ' ', text)
    text = text.strip()
    return text

class ResumeParser:
    def __init__(self, text):
        self.raw_text = text
        self.text = preprocess_text(text)

    def extract_skills(self):
        if not self.text:
            return []
        
        extracted_skills = set()
        text_lower = self.text.lower()
        
        for skill in ALL_SKILLS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                extracted_skills.add(skill)
        
        tech_pattern = r'\b[A-Za-z][\w\+\#\.\-]{2,}\b'
        tech_matches = re.findall(tech_pattern, self.text)
        
        for match in tech_matches:
            if (len(match) > 2 and 
                match.lower() not in ['the', 'and', 'for', 'with', 'that', 'have', 'this'] and
                not match.isdigit()):
                extracted_skills.add(match.lower())
        
        skills_list = list(extracted_skills)
        skills_list.sort()
        return skills_list

def score_resume_by_skills(resume_text, job_skills):
    if not job_skills:
        return 0.0, set()
    
    job_skills_lower = [skill.lower() for skill in job_skills]
    
    resume_lower = preprocess_text(resume_text).lower()
    matched_skills = set()
    
    for skill in job_skills_lower:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_lower):
            matched_skills.add(skill)
    
    score = (len(matched_skills) / len(set(job_skills_lower))) * 100
    return round(score, 2), matched_skills

def score_resume_by_text_similarity(resume_text, job_description_text):
    if not resume_text or not job_description_text:
        return 0.0
    
    processed_resume = preprocess_text(resume_text)
    processed_job = preprocess_text(job_description_text)
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        vectors = vectorizer.fit_transform([processed_resume, processed_job])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(similarity * 100, 2)
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def calculate_final_resume_score(resume_text, job_description_text, skill_weight=0.7, text_weight=0.3):
    if not resume_text or not job_description_text:
        return {
            "Skill Match Score": 0.0,
            "Text Similarity Score": 0.0,
            "Final Resume Score": 0.0,
            "Matched Skills": set(),
            "Job Skills": [],
            "Resume Skills": []
        }
    
    job_parser = ResumeParser(job_description_text)
    job_skills = job_parser.extract_skills()
    
    resume_parser = ResumeParser(resume_text)
    resume_skills = resume_parser.extract_skills()
    
    skill_score, matched_skills = score_resume_by_skills(resume_text, job_skills)
    text_similarity_score = score_resume_by_text_similarity(resume_text, job_description_text)
    
    final_score = round((skill_score * skill_weight) + (text_similarity_score * text_weight), 2)
    
    return {
        "Skill Match Score": skill_score,
        "Text Similarity Score": text_similarity_score,
        "Final Resume Score": final_score,
        "Matched Skills": matched_skills,
        "Job Skills": job_skills,
        "Resume Skills": resume_skills
    }

def main():
    st.set_page_config(
        page_title="AI Resume Matcher",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 AI Resume and Job Description Matcher")
    st.write("Upload your resume and paste the job description to see how well they match!")

    col1, col2 = st.columns([1, 1])

    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    with col2:
        job_description_text = st.text_area("Paste Job Description", height=200)

    col_weights1, col_weights2 = st.columns([1, 1])
    with col_weights1:
        skill_weight = st.slider("Skill Match Weight", 0.0, 1.0, 0.7, 0.1)
    with col_weights2:
        text_weight = st.slider("Text Similarity Weight", 0.0, 1.0, 0.3, 0.1, 
                               help="These weights determine how much each factor contributes to the final score")

    if st.button("Analyze Match"):
        if not resume_file:
            st.error("Please upload a resume PDF file")
            return
        if not job_description_text:
            st.error("Please paste a job description")
            return

        with st.spinner("Analyzing resume and job description..."):
            resume_text = extract_text_from_pdf(resume_file)
            
            if not resume_text:
                st.error("Could not extract text from the PDF. Please try another file.")
                return
            
            result = calculate_final_resume_score(
                resume_text, 
                job_description_text,
                skill_weight,
                text_weight
            )

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Skill Match", f"{result['Skill Match Score']}%", 
                          help="Percentage of required skills found in your resume")
            with col2:
                st.metric("Content Similarity", f"{result['Text Similarity Score']}%",
                         help="Overall textual similarity between your resume and the job description")
            with col3:
                final_score = result['Final Resume Score']
                color = "green" if final_score >= 70 else "orange" if final_score >= 50 else "red"
                st.metric("Overall Match", f"{final_score}%",
                         help="Weighted average of skill match and content similarity")
                
            tab1, tab2, tab3 = st.tabs(["📊 Match Analysis", "🔍 Skills Breakdown", "💡 Improvement Tips"])
            
            with tab1:
                st.subheader("Match Analysis")
                
                st.write("#### ✅ Matched Skills")
                if result["Matched Skills"]:
                    for skill in sorted(result["Matched Skills"]):
                        st.markdown(f"- {skill}")
                else:
                    st.write("No skills matched. Try updating your resume with relevant keywords from the job description.")
                    
                missing_skills = set([s.lower() for s in result["Job Skills"]]) - set([s.lower() for s in result["Matched Skills"]])
                st.write("#### ❌ Missing Skills")
                if missing_skills:
                    for skill in sorted(missing_skills):
                        st.markdown(f"- {skill}")
                else:
                    st.write("Great job! Your resume covers all the skills mentioned in the job description.")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Job Description Skills")
                    if result["Job Skills"]:
                        for skill in sorted(result["Job Skills"]):
                            st.markdown(f"- {skill}")
                    else:
                        st.write("No skills identified in the job description.")
                
                with col2:
                    st.write("#### Your Resume Skills")
                    if result["Resume Skills"]:
                        for skill in sorted(result["Resume Skills"]):
                            st.markdown(f"- {skill}")
                    else:
                        st.write("No skills identified in your resume.")
            
            with tab3:
                st.subheader("Improvement Tips")
                
                if final_score < 50:
                    st.warning("Your resume needs significant improvement to match this job description.")
                elif final_score < 70:
                    st.info("Your resume is a moderate match for this position but could be improved.")
                else:
                    st.success("Your resume is a strong match for this position!")
                
                st.write("#### How to improve your match:")
                
                if missing_skills:
                    st.write("1. Add these missing keywords to your resume:")
                    st.write(", ".join(sorted(missing_skills)))
                
                if result["Text Similarity Score"] < 60:
                    st.write("2. Align your resume's terminology with the job description")
                    st.write("3. Use similar phrasing and industry terminology")
                
                st.write("4. Quantify your achievements with specific metrics")
                st.write("5. Tailor your professional summary to specifically address this role")

if __name__ == "__main__":
    main()
