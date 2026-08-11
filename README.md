# ResumeAI - Premium Single-Page Talent Acquisition Portal (v3.0)

ResumeAI is a high-performance, single-page web portal designed to parse, classify, and rank PDF resumes against custom job descriptions or standard roles. It operates **100% locally and offline**, ensuring complete data privacy and sub-100ms response times.

---

## 1. System Architecture

```
                       +---------------------------------------+
                       |    Single-Page Web UI (index.html)    |
                       +-------------------+---------------+---+
                                           |               ^
                                  POST /   |               | JSON
                            files & Form   v               | Response
                       +-------------------+---------------+---+
                       |        FastAPI Web Server (main.py)   |
                       +-------------------+-------------------+
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
     +-------------+-------------+                   +-------------+-------------+
     |   Local Text Extraction   |                   |    Local Machine Learning |
     |      (PyMuPDF / fitz)     |                   |  (scikit-learn Classifier)|
     +-------------+-------------+                   +-------------+-------------+
                   |                                               |
                   v                                               v
     +-------------+-------------+                   +-------------+-------------+
     | Name & Email Parser Regex |                   | TF-IDF & Cosine Similarity|
     | (Space-Tolerant LinkedIn) |                   |  (Custom Match Scoring)   |
     +---------------------------+                   +---------------------------+
```

### Tech Stack
* **Frontend**: HTML5, Vanilla CSS3 (Glassmorphic dark design, dynamic cogs loading micro-animations, slide-in result badges), FontAwesome v6.
* **Backend**: FastAPI (Python 3.13 macOS ARM64 compatible), Uvicorn.
* **Data Processing & ML**: PyMuPDF (`fitz`), scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `LabelEncoder`), Pandas, Numpy, Textstat.

---

## 2. Directory Structure

```
Resume Analyzer/
├── main.py                  # Main FastAPI server entry point and endpoint routers
├── resume_quality.py        # Local grammar heuristics & readability scoring functions
├── requirements.txt         # Declared python package dependencies
├── model.py                 # (Offline training helper for Logistic Regression classifier)
├── UpdatedResumeDataSet.csv # Kaggle resume classification training dataset
├── tfidf_reducer.pkl        # Compressed feature model for classification
├── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer model
├── label_encoder.pkl        # Encoded job classification categories target dictionary
├── resume_classifier.pkl    # Trained multi-class Logistic Regression ML model
├── uploads/                 # Temporary workspace storage for uploaded resumes
├── templates/
│   └── index.html           # SPA Dashboard HTML and async JavaScript controller
└── static/
    └── index.css            # Gold-and-carbon theme styling classes
```

---

## 3. Core Processing Pipeline

When a folder or batch of PDF resumes is uploaded:
1. **Raw Text Extraction**: The server reads the PDF files using PyMuPDF. Newlines are preserved to maintain structural formatting.
2. **Robust Name Parsing**:
   * Standard regex extractors fail on multi-column layouts because skills (like "Python Basic") are printed in the left column and get mistakenly read as names.
   * ResumeAI parses all text lines using a capitalization regex pattern (`^[A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*){1,3}$`) but filters out any lines containing domain/tech keywords (e.g. `Python`, `Java`, `SQL`, `Developer`, `Student`, `Fundamentals`) using an expanded exclusion list. This guarantees that candidate names (e.g., `MARIANNE SRUTHI`) are correctly extracted regardless of multi-column layouts.
3. **Space-Tolerant LinkedIn Link Extraction**:
   * Text extractors often insert spaces inside links (e.g., `linkedin.com/in/marianne- sruthi-prabhu/`). Standard link searchers truncate at the space.
   * ResumeAI searches for lines containing `linkedin.com/in/`, strips all internal whitespaces, and then applies the regex matching pattern. This reconstructs the correct URL, preventing broken links.
4. **Machine Learning Classification**: The text is cleaned and transformed using a fitted TF-IDF Vectorizer. A trained multi-class Logistic Regression model predicts the primary job category (e.g. *Data Science*, *Web Designing*, *Java Developer*).
5. **Compatibility Match Scoring**: Matches the custom job description against the resume text. A local character-and-word n-gram `TfidfVectorizer` computes the cosine similarity metric. This handles spelling variations and custom keyword scoring out-of-the-box.
6. **Dynamic Sorting & Grouping**: Resumes matching the job description are ranked and listed. If no resumes meet the compatibility thresholds, the application displays a grouped layout categorizing the resumes by their predicted roles.

---

## 4. How to Run Locally

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Launch the FastAPI development server using Uvicorn:
```bash
python3 -m uvicorn main:app --port 5000 --reload
```

### 3. Open the UI
Navigate to the local dashboard in your browser:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
