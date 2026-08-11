import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize LanguageTool
# Disable LanguageTool to prevent slow remote API timeouts and keep analysis instant
tool = None

# Define important sections in a resume
IMPORTANT_SECTIONS = {
    "skills": ["skills", "technical skills", "programming languages", "expertise"],
    "experience": ["experience", "work experience", "employment history", "projects"],
    "education": ["education", "qualifications", "academic background", "degree"]
}

# Define job-specific skills for all categories
JOB_SKILLS = {
    "Advocate": ["Legal Research", "Litigation", "Contract Law", "Court Proceedings", "Legal Writing", "Client Counseling"],
    "Arts": ["Painting", "Sketching", "Sculpting", "Illustration", "Adobe Photoshop", "Graphic Design"],
    "Automation Testing": ["Selenium", "JUnit", "TestNG", "Cypress", "LoadRunner", "JMeter", "Automated Test Scripts"],
    "Blockchain": ["Ethereum", "Solidity", "Hyperledger", "Smart Contracts", "Cryptography", "Decentralized Apps"],
    "Business Analyst": ["Requirement Gathering", "Process Modeling", "Data Analysis", "Stakeholder Management", "Agile", "JIRA"],
    "Civil Engineer": ["AutoCAD", "Structural Analysis", "Construction Management", "Surveying", "Revit", "Building Codes"],
    "Data Science": ["Python", "Machine Learning", "Deep Learning", "Pandas", "TensorFlow", "SQL", "Data Visualization"],
    "Database": ["SQL", "NoSQL", "MongoDB", "PostgreSQL", "Database Administration", "Performance Optimization"],
    "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Jenkins", "Terraform", "AWS", "Linux"],
    "DotNet Developer": [".NET Framework", "C#", "ASP.NET", "Entity Framework", "SQL Server", "MVC"],
    "ETL Developer": ["ETL Processes", "Informatica", "Talend", "Data Warehousing", "SSIS", "Data Pipelines"],
    "Electrical Engineering": ["Power Systems", "Circuit Design", "MATLAB", "Embedded Systems", "Renewable Energy"],
    "HR": ["Recruitment", "Employee Relations", "Payroll Management", "Performance Appraisal", "HR Policies"],
    "Hadoop": ["HDFS", "MapReduce", "Spark", "Hive", "Pig", "Big Data Analytics"],
    "Health and Fitness": ["Personal Training", "Nutrition", "Exercise Physiology", "Diet Planning", "Yoga"],
    "Java Developer": ["Java", "Spring Boot", "Hibernate", "REST APIs", "Microservices", "Maven"],
    "Mechanical Engineer": ["SolidWorks", "CAD", "Thermodynamics", "Product Design", "Manufacturing Processes"],
    "Network Security Engineer": ["Firewalls", "Penetration Testing", "SOC", "SIEM", "Intrusion Detection"],
    "Operations Manager": ["Supply Chain", "Process Optimization", "Project Management", "Lean Six Sigma"],
    "PMO": ["Project Management", "Agile", "Scrum", "Risk Management", "Stakeholder Communication"],
    "Python Developer": ["Python", "Django", "Flask", "REST APIs", "SQLAlchemy", "Pandas"],
    "SAP Developer": ["SAP ABAP", "SAP FICO", "SAP HANA", "SAP MM", "SAP BW"],
    "Sales": ["Lead Generation", "B2B Sales", "CRM", "Negotiation", "Market Research"],
    "Testing": ["Manual Testing", "Selenium", "JIRA", "Performance Testing", "Bug Tracking"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "UI/UX Design", "Responsive Web Design", "Figma"],
}

# Define strong action verbs to assess impactful resume writing
ACTION_VERBS = [
    "developed", "designed", "managed", "led", "implemented", "optimized",
    "created", "coordinated", "built", "executed", "supervised"
]

def check_resume_sections(resume_text):
    """Check if the resume contains key sections: Skills, Experience, Education."""
    score = sum(1 for section in IMPORTANT_SECTIONS if any(keyword in resume_text.lower() for keyword in IMPORTANT_SECTIONS[section]))
    return score

def calculate_tfidf_relevance(resume_text, job_category, all_resumes):
    """Calculate TF-IDF similarity between the resume and the given job category.

    NOTE: kept for backwards compatibility / single-resume calls only.
    For batches, use precompute_relevance_scores() instead -- calling this
    function once per resume in a loop re-fits a TF-IDF vectorizer over the
    ENTIRE corpus every single time, which is O(n^2) and will hang on any
    batch larger than a few hundred resumes.
    """
    if not all_resumes:
        return 0  # Avoid division by zero

    corpus = [resume_text] + all_resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vector = tfidf_matrix[0]
    category_vector = tfidf_matrix.mean(axis=0)

    similarity = (resume_vector @ category_vector.T).A1[0] if resume_vector.shape[1] > 0 else 0
    return round(similarity, 2)


def precompute_relevance_scores(all_resumes):
    """Fit ONE TF-IDF vectorizer over the whole corpus and return a
    per-resume relevance score list, in the same order as all_resumes.

    This replaces calling calculate_tfidf_relevance() once per resume inside
    a loop (which re-fits the vectorizer on the full corpus every time --
    O(n^2), and the actual cause of multi-minute hangs on large batches).
    Call this ONCE before processing resumes, then look up scores by index.
    """
    if not all_resumes:
        return []

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_resumes)
    except Exception as e:
        print(f"[resume_quality] precompute_relevance_scores failed: {e}")
        return [0.0] * len(all_resumes)

    if tfidf_matrix.shape[1] == 0:
        return [0.0] * len(all_resumes)

    mean_vector = tfidf_matrix.mean(axis=0)
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        similarity = (tfidf_matrix[i] @ mean_vector.T).A1[0]
        scores.append(round(float(similarity), 2))
    return scores

def check_grammar_errors(resume_text):
    """Count grammar and spelling mistakes using LanguageTool."""
    if tool:
        try:
            matches = tool.check(resume_text)
            return len(matches)
        except Exception as e:
            print(f"Warning: LanguageTool grammar check failed - {e}")
            return 0
    return 0  # If LanguageTool fails, assume no grammar errors

def check_job_skills(resume_text, job_category):
    """Check if the resume contains job-specific keywords."""
    skills_required = JOB_SKILLS.get(job_category, [])
    skills_found = [skill for skill in skills_required if skill.lower() in resume_text.lower()]

    match_percentage = round((len(skills_found) / len(skills_required)) * 100, 1) if skills_required else 0
    return match_percentage, skills_found

def check_action_verbs(resume_text):
    """Check if the resume contains strong action verbs."""
    verbs_found = [verb for verb in ACTION_VERBS if verb in resume_text.lower()]
    return len(verbs_found), verbs_found

def count_syllables_word(word):
    """Heuristic vowel-based syllable counter. Optimized for speed by avoiding regex."""
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(1, count)

def readability_score(resume_text):
    """Calculate the readability score of the resume using pure Python stats to avoid NLTK downloads."""
    words = re.findall(r'[a-zA-Z]+', resume_text)
    sentences = re.split(r'[.!?]+', resume_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    num_words = len(words)
    num_sentences = len(sentences)
    
    if num_words == 0 or num_sentences == 0:
        return "Unknown"
        
    num_syllables = sum(count_syllables_word(w) for w in words)
    
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    
    score = 206.835 - 1.015 * asl - 84.6 * asw
    
    if score > 60:
        return "Highly Readable"
    elif score > 30:
        return "Moderate Readability"
    else:
        return "Hard to Read"

def assess_resume_quality(resume_text, job_category, all_resumes=None, precomputed_tfidf_score=None):
    """Assess resume quality based on sections, TF-IDF relevance, job skills, grammar, readability, and action verbs.

    Pass precomputed_tfidf_score (from precompute_relevance_scores(), computed
    once for the whole batch) to avoid an expensive per-resume TF-IDF refit.
    Falls back to the old slow per-call behavior only if it's not provided.
    """

    section_score = check_resume_sections(resume_text)
    skill_match, matched_skills = check_job_skills(resume_text, job_category)
    if precomputed_tfidf_score is not None:
        tfidf_score = precomputed_tfidf_score
    else:
        tfidf_score = calculate_tfidf_relevance(resume_text, job_category, all_resumes)
    grammar_errors = check_grammar_errors(resume_text)
    action_verbs_count, action_verbs_used = check_action_verbs(resume_text)
    readability = readability_score(resume_text)

    # Assign weights to each factor
    weighted_score = (
        (section_score * 3) +
        (tfidf_score * 10) +
        (skill_match * 0.5) +
        (action_verbs_count * 2) -
        (grammar_errors * 0.3)
    )

    # Assign quality rating
    if weighted_score > 18:
        return "⭐⭐⭐⭐⭐ (Outstanding Resume)"
    elif weighted_score > 12:
        return "⭐⭐⭐⭐ (Good Resume)"
    elif weighted_score > 6:
        return "⭐⭐⭐ (Decent Resume)"
    else:
        return "⭐ (Needs Improvement)"
