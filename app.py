from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pdfplumber
import os
import datetime

app = Flask(__name__)

# 🔹 ENTER YOUR GOOGLE GEMINI API KEY BELOW
GEMINI_API_KEY = " "
genai.configure(api_key=GEMINI_API_KEY)

# Log files
CONSULTATION_LOG = "logs/medical_consultation_log.txt"
PATHOLOGY_LOG = "logs/pathology_analysis_log.txt"

# Ensure uploads and logs directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# List of valid medical terms (symptoms, diseases, conditions)
VALID_MEDICAL_TERMS = {
    "fever", "cough", "headache", "diabetes", "hypertension", "asthma", "flu", "cold", 
    "migraine", "pneumonia", "covid-19", "allergy", "infection", "bronchitis", 
    "arthritis", "anemia", "cancer", "depression", "anxiety", "stroke", 
    "heart attack", "obesity", "hair loss", "nausea", "vomiting", "diarrhea", 
    "constipation", "fatigue", "dizziness", "fainting", "shortness of breath", 
    "chest pain", "abdominal pain", "joint pain", "muscle pain", "back pain", 
    "skin rash", "swelling", "numbness", "weakness", "blurry vision", 
    "ear infection", "sinusitis", "gastritis", "ulcer", "food poisoning", 
    "kidney disease", "liver disease", "thyroid disorder", "autoimmune disease", 
    "seizures", "paralysis", "mental confusion", "loss of appetite", 
    "weight loss", "weight gain", "high cholesterol", "sleep apnea", "insomnia", 
    "acid reflux", "irritable bowel syndrome", "chronic pain", "dehydration", 
    "bleeding", "heat stroke", "meningitis", "tuberculosis", "hepatitis", 
    "pancreatitis", "lupus", "multiple sclerosis", "Parkinson’s disease", 
    "Alzheimer’s disease", "epilepsy", "blood clot", "varicose veins", 
    "immune deficiency", "psoriasis", "eczema", "hives", "sunburn", 
    "frostbite", "fracture", "sprain", "tendonitis", "concussion", 
    "brain tumor", "leukemia", "lymphoma", "melanoma"
}


class AIMedicalAssistant:
    """AI Medical Assistant for consultations and pathology report analysis."""

    def __init__(self):
        self.conversation_history = []  # Stores chat history
        self.symptom_responses = {}  # Caches AI responses to avoid repeated API calls
        self.current_symptom = None  # Tracks the current symptom being discussed
        self.stage = 0  # Tracks the stage of consultation
        self.current_question_index = 0  # Tracks the current question in the follow-up sequence

    def log_conversation(self, entry):
        """Logs conversation to a file."""
        with open(CONSULTATION_LOG, "a", encoding="utf-8") as file:
            file.write(f"{datetime.datetime.now()} - {entry}\n")

    def generate_ai_response(self, user_input):
        """Generates AI response using Google Gemini API."""
        if user_input in self.symptom_responses:
            return self.symptom_responses[user_input]
        
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_input)
            ai_response = response.text.strip()
            self.symptom_responses[user_input] = ai_response
            return ai_response
        except Exception as e:
            return f"Error: {e}"

    def structured_response(self, symptom):
        """Generates structured medical advice for a given symptom."""
        query = (
            f"Provide a structured response for the symptom: {symptom}. "
            "Include Definition, Causes, Symptoms, Risk Factors, Precautions, and When to Consult a Doctor. "
            "Also, suggest which specialist doctor to visit."
        )
        return self.generate_ai_response(query)

    def get_additional_questions(self):
        """Returns a list of additional follow-up questions."""
        return [
            "Do you have any other symptoms associated with this?",
            "Have you taken any medication or treatment for this?",
            "Have you experienced this before?",
            "Does it get worse under certain conditions?",
            "Are there any other health issues you have that might be related?"
        ]

    def start_consultation(self, user_input):
        """Runs the AI medical consultation assistant."""
        if self.stage == 0:
            if user_input.lower() not in VALID_MEDICAL_TERMS:
                return "AIMCA: Please enter a valid medical symptom or condition."
            self.current_symptom = user_input
            self.stage = 1
            return "AIMCA: How long have you had this symptom?"
        
        elif self.stage == 1:
            self.stage = 2
            return "AIMCA: On a scale of 1-10, how severe is it?"
        
        elif self.stage == 2:
            additional_questions = self.get_additional_questions()
            if self.current_question_index < len(additional_questions):
                question = additional_questions[self.current_question_index]
                self.current_question_index += 1
                return f"AIMCA: {question}"
            else:
                self.current_question_index = 0
                self.stage = 3
                return "AIMCA: Would you like to know precautions and when to see a doctor? (yes/no)"
        
        elif self.stage == 3:
            if user_input.lower() == "yes":
                response = self.structured_response(self.current_symptom)
                self.stage = 0
                return f"AIMCA: {response}"
            else:
                self.stage = 0
                return "AIMCA: Thank you for using AIMCA!"

class PathologyReportAnalyzer:
    """Extracts, analyzes, and logs pathology reports using AI."""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.extracted_text = ""
    
    def extract_text_from_pdf(self):
        """Extracts text from a pathology report PDF."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        self.extracted_text += text + "\n"
            return self.extracted_text if self.extracted_text else "Error: No readable text found."
        except Exception as e:
            return f"Error extracting text: {e}"

    def analyze_report(self):
        """Uses AI to analyze the pathology report."""
        if not self.extracted_text:
            return "No text extracted from the PDF."
        
        prompt = (
            "Analyze the following pathology report and provide a structured response including:\n"
            "1. Identified abnormalities\n"
            "2. Possible risk factors\n"
            "3. Recommended next steps\n"
            "4. Suggested specialist doctor to consult\n"
            "5. At-home remedies or lifestyle changes (if applicable)\n\n"
            f"Report:\n{self.extracted_text}"
        )
        
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error analyzing report: {e}"

    def log_analysis(self, analysis_result):
        """Logs analysis results."""
        with open(PATHOLOGY_LOG, "a", encoding="utf-8") as file:
            file.write(f"{datetime.datetime.now()}\n{analysis_result}\n{'='*80}\n")

    def run_analysis(self):
        """Executes the pathology report analysis."""
        print("Extracting text from pathology report...")
        if "Error" in (text := self.extract_text_from_pdf()):
            print(text)
            return
        print("Analyzing report with AI...")
        analysis_result = self.analyze_report()
        print("Logging results...")
        self.log_analysis(analysis_result)
        return analysis_result

# Global instances
assistant = AIMedicalAssistant()
analyzer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = assistant.start_consultation(user_input)
    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    global analyzer
    analyzer = PathologyReportAnalyzer(file_path)
    analysis_result = analyzer.run_analysis()
    return jsonify({'response': analysis_result})

if __name__ == '__main__':
    app.run(debug=True)
