import streamlit as st
from transformers import pipeline
import pdfkit
import os
from jinja2 import Environment, FileSystemLoader

# --- Setup ---
st.set_page_config(page_title="Resume Builder Using Transformers", layout="centered")

@st.cache_resource
def load_flan_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

generator = load_flan_model()

# Configure pdfkit path (Windows users should update this if different)
path_wkhtmltopdf = r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# --- AI Utilities ---
def polish_experience(raw_text):
    prompt = f"""
    Improve the following resume bullet point by making it more professional, action-oriented, and quantifiable. Use strong verbs, specify tools/technologies, and add measurable outcomes when possible.

    Examples:
    Original: built a website  
    Polished: Developed and deployed a responsive company website using React and Node.js, improving user engagement by 40%.

    Original: did internship at google in data  
    Polished: Completed a data analytics internship at Google, creating automated dashboards in Python that streamlined reporting for 5+ teams.

    Original: led a team of 4 in a web dev project  
    Polished: Led a team of 4 developers to build a full-stack web application using React and Flask, reducing delivery time by 25%.

    Original: worked on a chatbot project for a few months  
    Polished: Designed and implemented a customer service chatbot using Python and Dialogflow, decreasing average support resolution time by 30%.

    Original: {raw_text}  
    Polished:"""

    result = generator(prompt, max_length=120, temperature=0.9, top_p=0.95, do_sample=True)[0]['generated_text']
    polished = result.strip().split("\n")[0].strip("\u2022- ").strip()
    return polished

def generate_summary(name, role, experience, skills, projects, bio):
    prompt = f"""
    Write a concise and confident resume summary based on the following:
    Name: {name}
    Role: {role}
    Experience: {experience} years
    Key Skills: {skills}
    Projects: {projects}
    Bio: {bio}
    """
    result = generator(prompt, max_length=256, do_sample=True, temperature=0.9)[0]['generated_text']
    return result.strip()

def generate_bio(name, role, experience, skills, projects):
    prompt = f"""
    Write a professional third-person bio for a resume.

    Name: {name}
    Role: {role}
    Years of Experience: {experience}
    Skills: {skills}
    Key Projects: {projects}

    Format:
    - Start with a confident one-line introduction mentioning the name, role, and experience.
    - Summarize core strengths, domain expertise, and technical skills.
    - Highlight notable achievements or project contributions.
    - Avoid personal interests or hobbies.
    - Tone: Formal, professional, and concise.

    Resume Bio:
    """
    result = generator(prompt, max_length=300, temperature=0.7)[0]['generated_text']
    return result.strip()


def suggest_skills(bio, role):
    prompt = f"""
    Based on the following bio of an {role}, suggest only technical and domain-relevant skills. Avoid vague terms like "coding" or "sys". Focus on AI/ML/NLP/Data/Software skills only.

    Bio: {bio}
    Suggested Skills:
    """
    result = generator(prompt, max_length=100, temperature=0.9, top_p=0.95, do_sample=True)[0]['generated_text']
    return result.strip().replace("\n", ", ")

# --- PDF Creation ---
def create_pdf(name, role, experience, summary, experience_list, skills, bio, template_name="minimal.html"):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template(template_name)

    html_content = template.render(
        name=name,
        role=role,
        experience=experience,
        summary=summary,
        experiences=experience_list,
        skills=skills.split(",") if isinstance(skills, str) else skills,
        bio=bio
    )

    pdf = pdfkit.from_string(html_content, False, configuration=config)
    return pdf

# --- UI ---
st.title("Resume Builder Using Transformers")
st.write("Write Your Details to generate a resume")

tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Details", "💼 Projects/Experience", "✨ Auto-Fill from Bio", "📈 Generate & Export"])

# --- Tab 1: Personal Info ---
with tab1:
    st.header("Basic Information")
    name = st.text_input("Full Name")
    role = st.text_input("Target Role")
    experience = st.slider("Years of Experience", 0, 40, 2)
    skills = st.text_area("Key Skills", placeholder="e.g. Python, Deep Learning, NLP")

    if st.button("🔄 Suggest Skills", key="suggest_skills_button"):
        if name and role and experience:
            with st.spinner("Generating skill suggestions..."):
                temp_bio = generate_bio(name, role, experience, skills, "")
                suggestions = suggest_skills(temp_bio, role)
                st.success("✅ Suggested Skills:")
                st.markdown(f"**{suggestions}**")
        else:
            st.warning("Please fill in all fields above.")

# --- Tab 2: Projects / Experience ---
with tab2:
    st.header("Enhance Your Projects or Experience")
    raw_input = st.text_area("Paste vague bullet points (one per line)")
    enhance = st.button("✨ Enhance All Bullets")
    if "enhanced_experiences" not in st.session_state:
        st.session_state.enhanced_experiences = []

    if enhance:
        if raw_input.strip():
            st.session_state.enhanced_experiences = []
            raw_bullets = [line.strip("-• ").strip() for line in raw_input.splitlines() if line.strip()]
            with st.spinner("Enhancing..."):
                for bullet in raw_bullets:
                    st.session_state.enhanced_experiences.append(polish_experience(bullet))
            st.success("✅ Enhanced Bullet Points:")
            for e in st.session_state.enhanced_experiences:
                st.markdown(f"• {e}")
        else:
            st.warning("Please enter some bullet points.")

# --- Tab 3: Auto Bio ---
with tab3:
    st.header("Generate Your Professional Bio")
    bio_button = st.button("🔄 Generate Bio")
    if bio_button:
        if not name or not role or not skills:
            st.warning("Please complete the Personal Info tab.")
        else:
            with st.spinner("Generating bio..."):
                projects = "\n".join(st.session_state.get("enhanced_experiences", []))
                bio = generate_bio(name, role, experience, skills, projects)
                st.session_state.generated_bio = bio
                st.success("✅ Bio Generated:")
                st.write(bio)

    if st.session_state.get("generated_bio"):
        if st.button("🔄 Suggest Skills Based on Bio"):
            with st.spinner("Generating..."):
                suggestions = suggest_skills(st.session_state.generated_bio, role)
                st.success("✅ Suggested Skills:")
                st.markdown(f"**{suggestions}**")

# --- Tab 4: Final Generation ---
with tab4:
    st.header("Generate Summary and Download")

    if st.button("🔄 Generate Summary"):
        if not name or not role or not skills:
            st.warning("Please fill all required fields.")
        else:
            with st.spinner("Generating summary..."):
                projects = "\n".join(st.session_state.get("enhanced_experiences", []))
                bio = st.session_state.get("generated_bio", "")
                summary = generate_summary(name, role, experience, skills, projects, bio)
                st.session_state.generated_summary = summary
                st.success("✅ Resume Summary:")
                st.write(summary)

    if st.session_state.get("generated_summary"):
        st.markdown("---")
        template_choice = st.selectbox("Choose Resume Template", ["minimal.html", "professional.html", "creative.html"])
        if st.button("📄 Download Resume PDF"):
            pdf_data = create_pdf(
                name,
                role,
                experience,
                st.session_state.generated_summary,
                st.session_state.get("enhanced_experiences", []),
                skills,
                st.session_state.get("generated_bio", ""),
                template_choice
            )
            st.download_button("📄 Download Resume PDF", data=pdf_data, file_name="resume.pdf", mime="application/pdf")

        st.download_button("🔍 Download Summary TXT", st.session_state.generated_summary, file_name="resume_summary.txt")