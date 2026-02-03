import os
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes
import ollama

app = Flask(__name__)

# --- CONFIGURATION DEFAULTS ---
# These act as fallbacks if the UI sends empty values
DEFAULT_TESSERACT = r'D:\DCS\Crop Prediction\Tesseract-OCR\tesseract.exe'
DEFAULT_POPPLER = r'D:\D\Projects\chat_with_pdf\poppler-24.08.0\Library\bin'

def get_cv_text(file_bytes, filename, tess_path, pop_path):
    """
    Extracts text from PDF. Uses OCR if the PDF is scanned (image-based).
    """
    # Set Tesseract Path dynamically from UI input
    pytesseract.pytesseract.tesseract_cmd = tess_path
    
    text = ""
    try:
        # Method 1: Standard Text Extraction
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages[:4]: # Analyze first 4 pages
            content = page.extract_text()
            if content: text += content
        
        # Method 2: OCR Fallback (If text is empty/scanned)
        if len(text.strip()) < 50:
            # print(f"OCR Triggered for {filename}...") # Debug
            try:
                images = convert_from_bytes(file_bytes, poppler_path=pop_path, last_page=3)
                for img in images:
                    text += pytesseract.image_to_string(img)
            except Exception as e:
                print(f"OCR Warning for {filename}: {e}")
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return ""
    
    return text

def analyze_candidate(cv_text, role, skills, exp, model):
    """
    Sends the CV + Criteria to Ollama for analysis.
    """
    system_prompt = (
        "You are an Elite Technical Recruiter for a top Fortune 500 company. "
        "Your job is to strictly evaluate candidates based on provided criteria. "
        "SCORING RULES: "
        "1. Education Match (20%): Exact degree/field match. "
        "2. Experience (30%): Deduct points for low experience. "
        "3. Skills (30%): Check for required tools/languages. "
        "4. Relevance (20%): Domain fit. "
        "Output strictly valid JSON only: "
        '{"name": "Candidate Name", "score": 0-100, "status": "Shortlisted/Rejected", "summary": "One sentence specific feedback"}'
    )
    
    user_prompt = f"""
    TARGET ROLE: {role}
    REQUIRED SKILLS: {skills}
    REQUIRED EXPERIENCE: {exp}
    
    CANDIDATE CV CONTENT:
    {cv_text[:3500]}
    """
    
    try:
        response = ollama.generate(model=model, system=system_prompt, prompt=user_prompt, format="json")
        return json.loads(response['response'])
    except Exception as e:
        print(f"AI Error: {e}")
        return None

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Retrieve Configuration from Form
        tess_path = request.form.get('tesseract_path', DEFAULT_TESSERACT)
        pop_path = request.form.get('poppler_path', DEFAULT_POPPLER)
        model = request.form.get('model', 'mistral')
        
        # 2. Retrieve Files
        if 'criteria_file' not in request.files or 'cv_files' not in request.files:
            return jsonify({"status": "error", "message": "Missing files"})
            
        criteria_file = request.files['criteria_file']
        cv_files = request.files.getlist('cv_files')
        
        # 3. Retrieve Column Mappings
        col_role = request.form.get('col_role')
        col_skills = request.form.get('col_skills')
        col_exp = request.form.get('col_exp')

        # 4. Read Excel Criteria
        try:
            df = pd.read_excel(criteria_file)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid Excel file: {str(e)}"})

        results = []

        # 5. Processing Loop
        for cv in cv_files:
            # Extract Text
            cv_bytes = cv.read()
            cv_text = get_cv_text(cv_bytes, cv.filename, tess_path, pop_path)
            
            if not cv_text:
                continue # Skip empty/unreadable files

            # Match against EVERY row in the Excel sheet
            for _, row in df.iterrows():
                # Safe get columns based on user mapping
                role_val = str(row.get(col_role, "Unknown Role"))
                skills_val = str(row.get(col_skills, "General"))
                exp_val = str(row.get(col_exp, "Not Specified"))

                # Perform AI Analysis
                ai_res = analyze_candidate(cv_text, role_val, skills_val, exp_val, model)
                
                if ai_res:
                    results.append({
                        "candidate": ai_res.get("name", "Unknown Candidate"),
                        "role": role_val,
                        "score": ai_res.get("score", 0),
                        "status": ai_res.get("status", "Pending"),
                        "reason": ai_res.get("summary", "No feedback provided"),
                        "filename": cv.filename
                    })

        # 6. Sort Results (Highest Score First)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return jsonify({"status": "success", "data": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Create templates directory if not exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🚀 TalentScout Enterprise Server Running...")
    print("📂 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)