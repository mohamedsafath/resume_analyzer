from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import re
import uuid
import pickle
import pymupdf
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
import starlette.requests
import resume_quality

# Globally increase multipart form parsing limits to support 10,000+ files and fields
original_form = starlette.requests.Request.form
async def patched_form(self, *, max_files: int = 10000, max_fields: int = 10000, **kwargs):
    return await original_form(self, max_files=max_files, max_fields=max_fields, **kwargs)
starlette.requests.Request.form = patched_form

app = FastAPI(title="Resume Analyzer")

# Upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static and templates folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# Global in-memory task tracking for background asynchronous parsing
ANALYSIS_JOBS = {}

# Predefined keywords for local TF-IDF matcher fallback
JOB_KEYWORDS = {
    "Advocate": ["law", "court", "legal", "litigation", "case", "judge", "attorney"],
    "Arts": ["painting", "sculpture", "drawing", "illustration", "graphic design", "visual arts"],
    "Automation Testing": ["selenium", "test automation", "pytest", "cypress", "Jenkins", "regression testing"],
    "Blockchain": ["blockchain", "cryptocurrency", "Ethereum", "smart contracts", "NFT", "decentralized"],
    "Business Analyst": ["business analysis", "requirements gathering", "data modeling", "stakeholder management"],
    "Civil Engineer": ["construction", "structural design", "AutoCAD", "building codes", "surveying", "reinforced concrete"],
    "Data Science": ["machine learning", "data analysis", "python", "statistics", "AI", "deep learning", "big data"],
    "Database": ["SQL", "NoSQL", "PostgreSQL", "MongoDB", "database management", "DBMS"],
    "DevOps Engineer": ["docker", "kubernetes", "CI/CD", "Jenkins", "AWS", "terraform", "ansible"],
    "DotNet Developer": [".NET", "C#", "ASP.NET", "MVC", "Entity Framework", "SQL Server"],
    "ETL Developer": ["ETL", "data pipeline", "Informatica", "Talend", "SSIS", "data warehousing"],
    "Electrical Engineering": ["circuit design", "power systems", "electrical wiring", "microcontrollers", "PLC"],
    "HR": ["recruitment", "payroll", "employee engagement", "performance management", "HR policies"],
    "Hadoop": ["big data", "Hadoop", "Spark", "HDFS", "MapReduce", "Hive", "Pig"],
    "Health and Fitness": ["nutrition", "personal training", "exercise science", "wellness", "dietitian"],
    "Java Developer": ["java", "spring", "hibernate", "microservices", "J2EE", "multithreading"],
    "Mechanical Engineer": ["CAD", "solidworks", "thermodynamics", "manufacturing", "design", "mechanical systems"],
    "Network Security Engineer": ["cybersecurity", "firewalls", "intrusion detection", "VPN", "penetration testing"],
    "Operations Manager": ["operations", "supply chain", "project management", "logistics", "process optimization"],
    "PMO": ["project management", "PMO", "stakeholder communication", "risk management", "resource planning"],
    "Python Developer": ["python", "django", "flask", "pandas", "numpy", "machine learning"],
    "SAP Developer": ["SAP", "ABAP", "SAP HANA", "SAP ERP", "Fiori", "SAP BW"],
    "Sales": ["sales", "CRM", "lead generation", "B2B", "negotiation", "customer acquisition"],
    "Testing": ["automation", "manual testing", "selenium", "test cases", "bug tracking", "QA"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "UI/UX", "responsive design", "frontend", "wireframing"]
}

# Pydantic schema for structured output from Gemini LLM
class ResumeExtractionSchema(BaseModel):
    name: str = Field(description="The candidate's full name")
    email: str = Field(description="The email address of the candidate")
    linkedin: str = Field(description="The LinkedIn URL of the candidate")
    predicted_category: str = Field(description="The core job category or professional role (e.g. Data Science, Web Developer, Java Developer, Product Manager, etc.)")
    skills: List[str] = Field(description="List of key technical or professional skills")
    education: str = Field(description="Highest degree and school")
    quality_rating: str = Field(description="Rating: '⭐ (Needs Improvement)', '⭐⭐⭐ (Decent Resume)', '⭐⭐⭐⭐ (Good Resume)', or '⭐⭐⭐⭐⭐ (Outstanding Resume)'")
    match_score: float = Field(description="Score between 0.0 and 100.0 evaluating match with the target job description")
    reasoning: str = Field(description="A brief 1-sentence explanation for this match score")

class AnalyzeRequest(BaseModel):
    job_description: str
    category: str
    filenames: List[str]

def load_model_artifacts():
    """Loads local machine learning model artifacts."""
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("resume_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, label_encoder, model

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF, preserving newlines. Never raises -- returns
    empty string on failure so a single bad file can't crash the batch."""
    try:
        doc = pymupdf.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()
        text = text.strip()
        if len(text) < 30:
            # Very little/no extractable text usually means a scanned
            # (image-only) PDF. We flag this distinctly so it doesn't get
            # silently misclassified -- OCR fallback can be added here later.
            print(f"[extract] WARNING: near-empty text ({len(text)} chars) from {pdf_path} -- likely a scanned/image PDF")
        return text
    except Exception as e:
        print(f"[extract] FAILED to read {pdf_path}: {e}")
        return ""

def clean_filename_to_name(filename):
    """Helper to strip extensions and separators from a filename to make it a name."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[-_]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_name(resume_text, filename):
    """Local heuristic to extract candidate name with a robust filter."""
    exclusions = {
        # Common Resume Section Headers & metadata
        "curriculum", "vitae", "resume", "cv", "contact", "summary", "experience", 
        "education", "profile", "about", "skills", "projects", "languages", "known",
        "objective", "interests", "activities", "awards", "achievements", "links", 
        "phone", "email", "address", "gender", "nationality", "declaration", "date", "place",
        # Technical & Domain terms
        "python", "java", "sql", "c++", "javascript", "basic", "intermediate", "expert",
        "student", "intern", "university", "college", "school", "professional", "exposure",
        "certifications", "extracurricular", "fluent", "native", "learning", "hobbies",
        "engineering", "prompt", "development", "database", "management", "frontend", "backend",
        "cloud", "infrastructure", "machine", "learning", "feature", "optimization",
        "natural", "language", "processing", "nlp", "systems", "application", "applications",
        "software", "web", "design", "movie", "full", "stack", "full-stack", "networks",
        "cybersecurity", "foundations", "associate", "fundamentals", "bootcamp", "training",
        "science", "technology", "information", "computer", "hobbies", "seminar",
        # Generic Job Titles/Header terms to avoid picking up as name
        "advocate", "officer", "qualifications", "highlights", "service", "bilingual", 
        "clinical", "client", "coordinator", "manager", "representative", "specialist",
        "administrator", "executive", "assistant", "practice", "practitioner", "therapist",
        "counselor", "analyst", "consultant", "operator", "agent", "worker", "helper",
        "legal", "court", "case", "study", "studies", "history", "details", "personal",
        "career", "work", "job", "employment", "salaries", "salary", "pay", "billing"
    }
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    name_pattern = r"^[A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*){1,3}$"
    for line in lines:
        cleaned = re.sub(r'[,|•·\-()]+', ' ', line).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if re.match(name_pattern, cleaned):
            words = [w.lower() for w in cleaned.split()]
            if not any(w in exclusions for w in words) and not any(char.isdigit() for char in cleaned):
                return cleaned
    return clean_filename_to_name(filename)

def extract_linkedin_email(resume_text):
    """Local heuristic to extract LinkedIn profile and email with space tolerance."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, resume_text)
    email = emails[0] if emails else 'N/A'

    linkedin = 'N/A'
    for line in resume_text.split('\n'):
        if 'linkedin.com/in/' in line:
            # Strip spaces from this line to handle multi-column split characters
            cleaned_line = line.replace(' ', '').replace('\t', '')
            match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_\-\/]+', cleaned_line)
            if match:
                linkedin = match.group(0)
                break
                
    # Fallback to standard regex if line search failed
    if linkedin == 'N/A':
        linkedin_pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_\-\/]+"
        linkedin_urls = re.findall(linkedin_pattern, resume_text)
        linkedin = linkedin_urls[0] if linkedin_urls else 'N/A'

    if linkedin != 'N/A' and not linkedin.startswith(('http://', 'https://')):
        linkedin = 'https://' + linkedin

    return linkedin, email

def calculate_local_similarity(resume_text, job_description):
    """Calculates custom TF-IDF cosine similarity score between resume text and job description."""
    if not resume_text or not job_description:
        return 0.0
    
    # We use character and word n-grams to perform robust match overlap
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        similarity = 0.0
        
    return float(max(0.0, min(100.0, round(similarity * 100, 2))))

def predict_local_category(resume_text, vectorizer, model, label_encoder):
    """Uses local scikit-learn model to categorize resume."""
    cleaned_text = " ".join(resume_text.lower().split())
    tfidf_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_vector)
    return label_encoder.inverse_transform(prediction)[0]

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serves the main single-page web dashboard."""
    return templates.TemplateResponse(request, "index.html", {
        "job_categories": list(JOB_KEYWORDS.keys()),
        "job_keywords": JOB_KEYWORDS
    })

def process_resumes_async(job_id: str, temp_resumes: list, job_description: str, target_category: str):
    """Synchronous background worker executed in a threadpool to avoid blocking the main Uvicorn event loop."""
    try:
        # Prepare context for local TF-IDF quality assessment
        local_vectorizer = None
        local_label_encoder = None
        local_classifier = None
        all_resumes_text = []

        try:
            local_vectorizer, local_label_encoder, local_classifier = load_model_artifacts()
            for r in temp_resumes:
                all_resumes_text.append(extract_text_from_pdf(r["path"]))
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning: Failed to load local ML models - {e}")

        # Scale progress: text extraction is done (let's say 10% progress)
        ANALYSIS_JOBS[job_id]["progress"] = max(1, int(len(temp_resumes) * 0.1))

        # Fit the quality-scoring TF-IDF vectorizer ONCE over the whole batch
        corpus_relevance_scores = resume_quality.precompute_relevance_scores(all_resumes_text)

        # Precompute match scores in bulk using a single TF-IDF fit
        match_scores = []
        if all_resumes_text and job_description:
            try:
                match_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
                match_tfidf = match_vectorizer.fit_transform(all_resumes_text + [job_description])
                jd_vector = match_tfidf[-1]
                similarities = cosine_similarity(match_tfidf[:-1], jd_vector).flatten()
                match_scores = [float(max(0.0, min(100.0, round(sim * 100, 2)))) for sim in similarities]
            except Exception as e:
                print(f"Warning: Bulk similarity calculation failed - {e}")
                match_scores = [0.0] * len(all_resumes_text)
        else:
            match_scores = [0.0] * len(all_resumes_text)

        import gc
        gc.collect()

        resumes_data = []
        resumes_by_category = {}
        total_matches = 0

        # Process each resume locally
        for idx, r in enumerate(temp_resumes):
            try:
                precomputed_score = corpus_relevance_scores[idx] if idx < len(corpus_relevance_scores) else None
                precomputed_match = match_scores[idx] if idx < len(match_scores) else 0.0
                resume_info = run_local_pipeline(
                    r, job_description, all_resumes_text, idx,
                    local_vectorizer, local_label_encoder, local_classifier,
                    precomputed_score, precomputed_match
                )

                clean_target = target_category.lower().strip() if target_category else ""
                clean_cat = resume_info["predicted_category"].lower().strip()

                if clean_target:
                    # Match only if the predicted category exactly matches the user-selected category
                    is_match = (clean_cat == clean_target)
                else:
                    # Fallback to high similarity score threshold if no category is selected
                    is_match = (resume_info["match_score"] >= 30.0)

                resume_info["is_match"] = is_match
                resumes_data.append(resume_info)

                if is_match:
                    total_matches += 1

                # Group by category
                cat = resume_info["predicted_category"]
                if cat not in resumes_by_category:
                    resumes_by_category[cat] = []
                resumes_by_category[cat].append(resume_info)
            except Exception as e:
                print(f"[match] FAILED to process {r['filename']}: {e}")
                error_info = {
                    "filename": r["filename"],
                    "predicted_category": "Error",
                    "quality_rating": "⭐ (Needs Improvement)",
                    "name": clean_filename_to_name(r["filename"]),
                    "linkedin": "N/A",
                    "email": "N/A",
                    "match_score": 0.0,
                    "reasoning": f"Processing failed: {e}",
                    "skills": [],
                    "education": "N/A",
                    "is_match": False,
                }
                resumes_data.append(error_info)
                resumes_by_category.setdefault("Error", []).append(error_info)

            # Update progress dynamically in memory (scaled from 10% to 100%)
            if idx % 50 == 0 or idx == len(temp_resumes) - 1:
                start_offset = int(len(temp_resumes) * 0.1)
                remaining_weight = int((idx / len(temp_resumes)) * (len(temp_resumes) * 0.9))
                ANALYSIS_JOBS[job_id]["progress"] = max(1, start_offset + remaining_weight)

        # Sort results
        resumes_data.sort(key=lambda x: x["match_score"], reverse=True)
        for cat in resumes_by_category:
            resumes_by_category[cat].sort(key=lambda x: x["match_score"], reverse=True)

        ANALYSIS_JOBS[job_id]["status"] = "completed"
        ANALYSIS_JOBS[job_id]["results"] = {
            "resumes": resumes_data,
            "resumes_by_category": resumes_by_category,
            "job_description": job_description,
            "total_matches": total_matches
        }
        
        print(f"[async job] {job_id} FINISHED: total={len(temp_resumes)} matches={total_matches}")
        import gc
        gc.collect()

    except Exception as ex:
        print(f"[async job] {job_id} CRASHED: {ex}")
        ANALYSIS_JOBS[job_id]["status"] = "failed"
        ANALYSIS_JOBS[job_id]["error"] = str(ex)

@app.post("/api/upload")
async def upload_resumes(
    resumes: List[UploadFile] = File(...)
):
    """API endpoint to upload a batch of resumes. Saves them to the uploads folder using chunk streaming."""
    saved_files = []
    skipped_files = []
    
    for file in resumes:
        if not file.filename.lower().endswith(".pdf"):
            skipped_files.append(f"{file.filename} (not a .pdf)")
            continue
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                while chunk := await file.read(65536):
                    f.write(chunk)
            
            # Verify file size on disk
            if os.path.getsize(file_path) == 0:
                skipped_files.append(f"{file.filename} (empty file)")
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                continue

            saved_files.append(file.filename)
        except Exception as e:
            skipped_files.append(f"{file.filename} (save failed: {e})")

    return JSONResponse({
        "uploaded": saved_files,
        "skipped": skipped_files
    })

@app.post("/api/analyze")
async def analyze_resumes(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """API endpoint to initialize and queue an asynchronous analysis job using pre-uploaded filenames."""
    temp_resumes = []
    skipped_files = []
    
    for filename in request.filenames:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            temp_resumes.append({
                "filename": filename,
                "path": file_path
            })
        else:
            skipped_files.append(f"{filename} (not found on server)")

    if not temp_resumes:
        return JSONResponse(status_code=400, content={
            "error": "No valid PDF resumes found on the server to analyze.",
            "skipped_files": skipped_files
        })

    # Create job entry and queue background processing
    job_id = str(uuid.uuid4())
    ANALYSIS_JOBS[job_id] = {
        "status": "processing",
        "progress": 0,
        "total": len(temp_resumes),
        "results": None
    }

    background_tasks.add_task(process_resumes_async, job_id, temp_resumes, request.job_description, request.category)

    return JSONResponse({
        "job_id": job_id,
        "skipped_files": skipped_files
    })

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Endpoint for checking asynchronous parsing progress and fetching final results."""
    job = ANALYSIS_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "not_found", "error": "Job ID not found"})
    
    if job["status"] == "completed":
        return JSONResponse({
            "status": "completed",
            "results": job["results"]
        })
    elif job["status"] == "failed":
        return JSONResponse({
            "status": "failed",
            "error": job.get("error", "Unknown processing error")
        })
    else:
        return JSONResponse({
            "status": "processing",
            "progress": job["progress"],
            "total": job["total"]
        })

def run_local_pipeline(r, job_description, corpus, idx, vectorizer, label_encoder, classifier, precomputed_tfidf_score=None, precomputed_match_score=None):
    """Executes the local parser, ML prediction model, and local similarity calculations."""
    # Reuse the text already extracted into `corpus` instead of re-reading
    # and re-parsing the PDF a second time.
    text = corpus[idx] if idx < len(corpus) and corpus[idx] else extract_text_from_pdf(r["path"])
    name = extract_name(text, r["filename"])
    linkedin, email = extract_linkedin_email(text)
    
    # Cosine Similarity
    if precomputed_match_score is not None:
        match_score = precomputed_match_score
    else:
        match_score = calculate_local_similarity(text, job_description)
    
    # ML Classifications
    if vectorizer and classifier:
        try:
            predicted_category = predict_local_category(text, vectorizer, classifier, label_encoder)
        except Exception:
            predicted_category = "Data Science"
    else:
        predicted_category = "Data Science"

    # Quality Scorer -- uses the precomputed corpus-wide relevance score
    # instead of re-fitting a TF-IDF vectorizer over the whole corpus here.
    quality_rating = resume_quality.assess_resume_quality(
        text, predicted_category, corpus, precomputed_tfidf_score=precomputed_tfidf_score
    )

    # Key Skills heuristic parsing for local display
    skills = []
    if predicted_category in JOB_KEYWORDS:
        skills = [sk for sk in JOB_KEYWORDS[predicted_category] if sk.lower() in text.lower()]

    return {
        "filename": r["filename"],
        "predicted_category": predicted_category,
        "quality_rating": quality_rating,
        "name": name,
        "linkedin": linkedin,
        "email": email,
        "match_score": match_score,
        "reasoning": f"Scored locally using TF-IDF text vector similarity ({match_score}% vocabulary correlation).",
        "skills": skills,
        "education": "N/A (Local parsing fallback)"
    }
