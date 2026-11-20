# ==============================================================================
# HEALTHCARE COGNITIVE ASSISTANT (GOOGLE COLAB READY)
# Covers Requirements 1–10 and Automated Test Cases 1–5
# ==============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Download required NLTK packages
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ==============================================================================
# REQUIREMENT 1: LABELLED CORPUS (72 QUERIES ACROSS 6 INTENT CATEGORIES)
# ==============================================================================
corpus_data = [
    # 1. symptom_inquiry (12 queries)
    ("I have high fever and body ache symptoms since yesterday", "symptom_inquiry"),
    ("Why do I have a throbbing headache and dizziness symptom", "symptom_inquiry"),
    ("I am experiencing dry cough and sore throat pain", "symptom_inquiry"),
    ("What causes stomach ache acidity and burning pain after eating", "symptom_inquiry"),
    ("I feel dizzy weak and fatigued all day long", "symptom_inquiry"),
    ("There is an itchy red skin rash on my arm", "symptom_inquiry"),
    ("I have sharp lower back pain and muscle ache", "symptom_inquiry"),
    ("My nose is blocked with sneezing and sinus headache", "symptom_inquiry"),
    ("I am having nausea vomiting and stomach cramps", "symptom_inquiry"),
    ("Why are my joints swollen with morning pain and stiffness", "symptom_inquiry"),
    ("I have ear pain and mild fever since last night", "symptom_inquiry"),
    ("What causes persistent cough fatigue and throat irritation", "symptom_inquiry"),

    # 2. medication_guidance (12 queries)
    ("What is the safe dosage of paracetamol tablet for adults", "medication_guidance"),
    ("Can I take ibuprofen medicine with food for pain", "medication_guidance"),
    ("What are the common side effects of amoxicillin antibiotic medicine", "medication_guidance"),
    ("How should I store insulin medication pens properly", "medication_guidance"),
    ("Is cetirizine allergy tablet safe to take daily", "medication_guidance"),
    ("Can I take antacid medicine together with blood pressure pill", "medication_guidance"),
    ("What happens if I miss a dose of thyroid tablet", "medication_guidance"),
    ("Does cough syrup medication cause side effects like drowsiness", "medication_guidance"),
    ("Should I take vitamin D supplement capsule after meals", "medication_guidance"),
    ("How many days should I take my prescribed antibiotic dosage", "medication_guidance"),
    ("What are the side effects and dosage of paracetamol medicine", "medication_guidance"),
    ("Can I take painkiller medication on an empty stomach", "medication_guidance"),

    # 3. appointment_booking (12 queries)
    ("I want to book an appointment with a general physician doctor", "appointment_booking"),
    ("Schedule a doctor consultation appointment with a cardiologist", "appointment_booking"),
    ("Is a dermatologist specialist available for clinic appointment Friday", "appointment_booking"),
    ("I need to reschedule my doctor appointment to next week", "appointment_booking"),
    ("How can I cancel my upcoming orthopedic doctor appointment", "appointment_booking"),
    ("Book a pediatrician consultation appointment slot for my child", "appointment_booking"),
    ("Check doctor availability to book an ENT specialist visit", "appointment_booking"),
    ("I want to schedule a follow up consultation with neurologist", "appointment_booking"),
    ("Can I book an online video consultation appointment with doctor", "appointment_booking"),
    ("Reserve a morning appointment slot for physician checkup", "appointment_booking"),
    ("Help me book a clinic visit with a skin specialist doctor", "appointment_booking"),
    ("I would like to schedule a medical consultation tomorrow", "appointment_booking"),

    # 4. preventive_wellness (12 queries)
    ("What healthy diet foods boost immunity and wellness", "preventive_wellness"),
    ("How many hours of sleep are needed for healthy wellness", "preventive_wellness"),
    ("Suggest a daily exercise routine for heart health wellness", "preventive_wellness"),
    ("How much water should I drink daily for healthy hydration", "preventive_wellness"),
    ("What preventive vaccines are recommended for healthy adults", "preventive_wellness"),
    ("How can I reduce stress for better mental health wellness", "preventive_wellness"),
    ("Give healthy lifestyle tips to prevent high blood sugar", "preventive_wellness"),
    ("What balanced diet and exercise plan supports healthy weight", "preventive_wellness"),
    ("How often should I get a preventive wellness health checkup", "preventive_wellness"),
    ("What stretching exercise routine prevents back and neck strain", "preventive_wellness"),
    ("Share daily wellness habits for better sleep and nutrition", "preventive_wellness"),
    ("How does regular exercise and healthy diet prevent disease", "preventive_wellness"),

    # 5. diagnostic_lab_info (12 queries)
    ("Do I need fasting before a lipid profile blood test", "diagnostic_lab_info"),
    ("What does a complete blood count CBC lab test measure", "diagnostic_lab_info"),
    ("How should I prepare for an abdominal ultrasound diagnostic scan", "diagnostic_lab_info"),
    ("What is the normal range for HbA1c diabetes blood test", "diagnostic_lab_info"),
    ("How long do thyroid function blood test lab reports take", "diagnostic_lab_info"),
    ("Can I drink water before a fasting glucose blood test", "diagnostic_lab_info"),
    ("What parameters does a liver function lab test check", "diagnostic_lab_info"),
    ("Why did my doctor order a vitamin B12 blood test report", "diagnostic_lab_info"),
    ("Is fasting preparation needed before a chest X ray scan", "diagnostic_lab_info"),
    ("What does a kidney creatinine blood test result indicate", "diagnostic_lab_info"),
    ("Explain what cholesterol and lipid blood test results measure", "diagnostic_lab_info"),
    ("How to prepare for diagnostic pathology blood sample collection", "diagnostic_lab_info"),

    # 6. emergency_alert (12 queries)
    ("Emergency I have severe chest pain radiating to left arm", "emergency_alert"),
    ("Urgent emergency someone has sudden difficulty breathing and gasping", "emergency_alert"),
    ("Emergency signs of stroke slurred speech and sudden weakness", "emergency_alert"),
    ("Severe uncontrolled bleeding from deep wound need emergency help", "emergency_alert"),
    ("Emergency person collapsed and is unconscious not responding", "emergency_alert"),
    ("Severe allergic reaction throat swelling and difficulty breathing", "emergency_alert"),
    ("Emergency someone took a massive overdose of sleeping pills", "emergency_alert"),
    ("Sudden loss of vision and severe chest pressure emergency", "emergency_alert"),
    ("Child having continuous seizure emergency ambulance needed", "emergency_alert"),
    ("Crushing chest pain with cold sweats possible heart attack", "emergency_alert"),
    ("Urgent medical emergency patient is unconscious and not breathing", "emergency_alert"),
    ("Severe head injury with heavy bleeding and unconsciousness", "emergency_alert"),
]

df = pd.DataFrame(corpus_data, columns=["query", "intent"])

# ==============================================================================
# REQUIREMENT 2 & 3: HIERARCHICAL TAXONOMY & DOMAIN ONTOLOGY
# ==============================================================================
HEALTHCARE_TAXONOMY = {
    "1. Clinical Care & Triage": {
        "1.1 Symptom Assessment": ["Acute Symptoms", "Chronic & Recurring Symptoms"],
        "1.2 Emergency Triage": ["Cardiopulmonary Emergencies", "Neurological & Trauma Emergencies"],
    },
    "2. Therapeutics & Pharmacology": {
        "2.1 Medication Information": ["Dosage & Administration", "Side Effects & Drug Interactions"],
        "2.2 Treatment Protocols": ["Over-the-Counter Remedies", "Prescription Therapies"],
    },
    "3. Diagnostics & Laboratory Services": {
        "3.1 Pathology & Blood Tests": ["Hematology & Metabolic Panels", "Endocrine & Lipid Profiles"],
        "3.2 Medical Imaging & Preparation": ["Radiology & Ultrasound Scans", "Pre-Test Fasting & Preparation"],
    },
    "4. Healthcare Operations & Preventive Health": {
        "4.1 Patient Scheduling & Administration": ["Appointment Booking", "Consultation Rescheduling & Cancellation"],
        "4.2 Preventive Medicine & Wellness": ["Nutrition, Hydration & Fitness", "Immunization & Routine Screening"],
    },
}

# 10 Domain Concepts
DOMAIN_CONCEPTS = [
    "Patient", "Symptom", "Condition", "Medication", "DiagnosticTest",
    "Physician", "Appointment", "EmergencyProtocol", "PreventiveMeasure", "SideEffect",
]

# 12 Semantic Relationships among Domain Concepts
DOMAIN_RELATIONSHIPS = [
    ("Patient", "exhibits", "Symptom"),
    ("Symptom", "indicates", "Condition"),
    ("Symptom", "triggers", "EmergencyProtocol"),
    ("Condition", "treated_with", "Medication"),
    ("Condition", "diagnosed_by", "DiagnosticTest"),
    ("Condition", "prevented_by", "PreventiveMeasure"),
    ("Medication", "may_cause", "SideEffect"),
    ("Medication", "prescribed_by", "Physician"),
    ("Patient", "books", "Appointment"),
    ("Appointment", "scheduled_with", "Physician"),
    ("DiagnosticTest", "ordered_by", "Physician"),
    ("Patient", "advised_to_follow", "PreventiveMeasure"),
]

# ==============================================================================
# REQUIREMENT 4: NLP PREPROCESSING (LOWERCASE, TOKENIZE, STOPWORDS, LEMMATIZE)
# ==============================================================================
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = set(ENGLISH_STOP_WORDS)

# Retain clinically meaningful prepositions/negations
stop_words -= {"no", "not", "nor", "after", "before", "during", "when", "with", "without"}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    # 1. Lowercase conversion
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    # 2. Tokenization
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\b[a-z]+\b", text)

    # 3. Stopword removal & 4. Lemmatization
    cleaned_tokens = []
    for tok in tokens:
        if tok not in stop_words and len(tok) > 1:
            try:
                lemma = lemmatizer.lemmatize(tok)
            except LookupError:
                lemma = tok[:-1] if (tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3) else tok
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

df["processed_query"] = df["query"].apply(preprocess_text)

# ==============================================================================
# REQUIREMENT 5: TF-IDF FEATURE EXTRACTION & CLASSIFIER TRAINING
# ==============================================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["processed_query"],
    df["intent"],
    test_size=0.25,
    random_state=42,
    stratify=df["intent"],
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=600)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# ==============================================================================
# REQUIREMENT 6: MODEL EVALUATION (ACCURACY, PRECISION, RECALL, F1, CONFUSION MATRIX)
# ==============================================================================
y_pred = classifier.predict(X_test)
labels = sorted(df["intent"].unique())

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("=" * 72)
print("MODEL EVALUATION REPORT")
print("=" * 72)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("Confusion Matrix:\n", cm_df)

# Plot Confusion Matrix Heatmap in Google Colab
plt.figure(figsize=(8, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Healthcare Intent Classifier - Confusion Matrix")
plt.ylabel("Actual Intent")
plt.xlabel("Predicted Intent")
plt.tight_layout()
plt.show()

# ==============================================================================
# REQUIREMENT 7: DOMAIN-SPECIFIC KNOWLEDGE BASE
# ==============================================================================
KNOWLEDGE_BASE = {
    "symptom_inquiry": {
        "taxonomy_path": "1. Clinical Care & Triage -> 1.1 Symptom Assessment",
        "concepts": ["Patient", "Symptom", "Condition", "Physician"],
        "default_response": (
            "Mild acute symptoms such as low-grade fever, headache, fatigue, or sore throat "
            "often result from viral infections, dehydration, or fatigue. Rest, hydrate adequately, "
            "and monitor your temperature."
        ),
        "topic_snippets": {
            "fever": "For mild fever, rest and drink plenty of fluids. Seek medical attention if fever exceeds 102°F (38.9°C) or persists beyond 3 days.",
            "headache": "Headaches commonly stem from dehydration, tension, or eye strain. Rest in a quiet environment and hydrate.",
            "throat": "Sore throat is frequently viral. Warm saltwater gargles and warm fluids help soothe throat irritation.",
            "cough": "A dry cough can be eased with steam inhalation, warm liquids, and avoiding irritants.",
            "acidity": "Acidity and heartburn improve with smaller meals, avoiding spicy/oily foods late at night, and staying upright after eating.",
            "rash": "Avoid scratching an itchy skin rash; consult a dermatologist if it spreads or blisters.",
        },
    },
    "medication_guidance": {
        "taxonomy_path": "2. Therapeutics & Pharmacology -> 2.1 Medication Information",
        "concepts": ["Medication", "Condition", "SideEffect", "Physician"],
        "default_response": (
            "Always follow the prescribed dosage schedule on your medicine label and never exceed the maximum daily limit."
        ),
        "topic_snippets": {
            "paracetamol": "Paracetamol (acetaminophen) relieves fever and mild-to-moderate pain. Standard adult dosing is typically 500–650 mg every 6–8 hours (max 4,000 mg/24 hrs); rare side effects include allergic rash or liver strain if overdosed.",
            "ibuprofen": "Ibuprofen is an NSAID for pain and inflammation; take it with food or milk to prevent stomach irritation.",
            "antibiotic": "Complete the full prescribed antibiotic course even if symptoms improve early, to prevent bacterial resistance.",
            "insulin": "Store unopened insulin pens refrigerated at 2°C–8°C and keep in-use pens away from heat and direct sunlight.",
        },
    },
    "appointment_booking": {
        "taxonomy_path": "4. Healthcare Operations & Preventive Health -> 4.1 Patient Scheduling & Administration",
        "concepts": ["Patient", "Appointment", "Physician"],
        "default_response": (
            "Appointment Scheduling Desk: In-clinic and online tele-consultation slots are available Monday to Saturday "
            "(Morning: 9:00 AM – 1:00 PM | Evening: 4:00 PM – 8:00 PM). Please share your preferred date and time to confirm."
        ),
        "topic_snippets": {
            "cardiologist": "Cardiology Consultation Slots: Available Mon, Wed, and Fri (10:00 AM – 1:00 PM).",
            "dermatologist": "Dermatology Consultation Slots: Available Tue, Thu, and Fri (4:00 PM – 7:00 PM).",
            "reschedule": "To reschedule, provide your booking reference ID along with your new preferred date and time.",
            "cancel": "Appointments can be cancelled without fees up to 2 hours prior to the scheduled consultation.",
        },
    },
    "preventive_wellness": {
        "taxonomy_path": "4. Healthcare Operations & Preventive Health -> 4.2 Preventive Medicine & Wellness",
        "concepts": ["Patient", "PreventiveMeasure", "Condition"],
        "default_response": (
            "Core preventive wellness pillars include a balanced diet rich in whole foods, 150 minutes of weekly aerobic exercise, "
            "7–8 hours of quality sleep, adequate hydration, and annual preventive health checkups."
        ),
        "topic_snippets": {
            "heart": "Heart-healthy habits include 30 minutes of brisk walking/cardio 5 days a week, low-sodium balanced meals, and regular blood pressure checks.",
            "diet": "Prioritize vegetables, fruits, whole grains, lean proteins, and healthy fats while limiting added sugars and trans fats.",
            "exercise": "Combine moderate aerobic activity (walking, cycling, swimming) with twice-weekly strength and flexibility exercises.",
            "sleep": "Maintain a consistent sleep schedule of 7–9 hours per night and reduce screen exposure before bedtime.",
        },
    },
    "diagnostic_lab_info": {
        "taxonomy_path": "3. Diagnostics & Laboratory Services -> 3.1 Pathology & Blood Tests",
        "concepts": ["DiagnosticTest", "Condition", "Physician"],
        "default_response": (
            "Diagnostic pathology and imaging tests assist physicians in clinical evaluation. Routine lab reports are usually delivered within 6–24 hours."
        ),
        "topic_snippets": {
            "cbc": "A Complete Blood Count (CBC) measures red blood cells, white blood cells, hemoglobin, and platelets to screen for anemia, infection, and blood disorders.",
            "lipid": "A Lipid Profile evaluates total cholesterol, LDL, HDL, and triglycerides to assess cardiovascular risk.",
            "hba1c": "HbA1c measures average blood glucose over 2–3 months (Normal: <5.7%, Prediabetes: 5.7%–6.4%, Diabetes: >=6.5%).",
            "thyroid": "A Thyroid Function Test checks TSH, Free T3, and Free T4 levels to assess thyroid gland activity.",
            "ultrasound": "Abdominal ultrasounds generally require 6–8 hours of fasting prior to the scan.",
        },
    },
    "emergency_alert": {
        "taxonomy_path": "1. Clinical Care & Triage -> 1.2 Emergency Triage",
        "concepts": ["Patient", "Symptom", "EmergencyProtocol"],
        "default_response": (
            "CRITICAL MEDICAL EMERGENCY: Seek immediate emergency medical attention! "
            "Call your local emergency helpline right away (112 / 108 in India, 911 in the US) "
            "or go to the nearest hospital Emergency Department immediately. Do not drive yourself."
        ),
        "topic_snippets": {},
    },
}

# ==============================================================================
# REQUIREMENT 8 & 9: RULE ENGINE + INTEGRATED COGNITIVE ASSISTANT
# ==============================================================================
class HealthcareCognitiveAssistant:
    def __init__(self, vectorizer, classifier, knowledge_base, taxonomy, concepts, relationships):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.kb = knowledge_base
        self.taxonomy = taxonomy
        self.concepts = concepts
        self.relationships = relationships
        self.confidence_threshold = 0.35

        self.emergency_keywords = [
            "chest pain", "crushing pressure", "difficulty breathing", "can't breathe",
            "cannot breathe", "gasping", "unconscious", "collapsed", "severe bleeding",
            "uncontrolled bleeding", "slurred speech", "stroke", "heart attack",
            "overdose", "seizure", "anaphylaxis", "throat swelling", "loss of vision",
        ]

        self.unsupported_patterns = [
            "surgical stitches", "stitches at home", "perform surgery", "operation at home",
            "veterinary", "dog", "cat", "pet", "narcotic", "controlled substance",
            "malpractice", "legal", "programming", "weather", "stock market",
        ]

        self.domain_anchors = {
            "fever", "pain", "headache", "cough", "throat", "rash", "symptom", "sick",
            "medicine", "medication", "tablet", "pill", "dose", "dosage", "paracetamol",
            "ibuprofen", "antibiotic", "insulin", "syrup", "vitamin", "drug", "side",
            "appointment", "book", "schedule", "doctor", "physician", "specialist",
            "clinic", "consultation", "reschedule", "cancel", "visit", "dermatologist",
            "cardiologist", "diet", "sleep", "exercise", "water", "hydrate", "vaccine",
            "stress", "wellness", "health", "healthy", "heart", "blood", "test", "cbc",
            "lipid", "scan", "ultrasound", "xray", "ray", "hba1c", "thyroid", "glucose",
            "fast", "fasting", "lab", "report", "emergency", "bleeding", "chest",
            "breathing", "unconscious", "overdose", "seizure", "swelling", "acidity",
            "nausea", "dizzy", "fatigue", "joint", "back", "allergy", "scratchy",
        }

    def _get_ontology_triples(self, active_concepts):
        return [
            f"({s} --[{p}]--> {o})"
            for (s, p, o) in self.relationships
            if s in active_concepts or o in active_concepts
        ][:3]

    def _reason_and_retrieve(self, raw_query: str, processed_query: str, predicted_intent: str, confidence: float):
        q_lower = raw_query.lower()
        tokens = set(processed_query.split())
        raw_words = set(re.findall(r"\b[a-z]+\b", q_lower))
        triggered_rules = []

        # RULE 1: Emergency Escalation (Highest Priority)
        if any(kw in q_lower for kw in self.emergency_keywords) or predicted_intent == "emergency_alert":
            triggered_rules.append("Rule 1: Emergency Escalation Protocol")
            kb_entry = self.kb["emergency_alert"]
            return {
                "status": "EMERGENCY_ESCALATION",
                "intent": "emergency_alert",
                "confidence": max(confidence, 0.99) if any(kw in q_lower for kw in self.emergency_keywords) else confidence,
                "triggered_rules": triggered_rules,
                "taxonomy": kb_entry["taxonomy_path"],
                "ontology_links": self._get_ontology_triples(kb_entry["concepts"]),
                "response": kb_entry["default_response"],
            }

        # RULE 2: Unsupported / Out-of-Scope Query Handling
        has_unsupported = any(pat in q_lower for pat in self.unsupported_patterns)
        has_domain_overlap = len((tokens | raw_words) & self.domain_anchors) > 0
        if has_unsupported or not has_domain_overlap:
            triggered_rules.append("Rule 2: Unsupported / Out-of-Scope Query Handling")
            return {
                "status": "UNSUPPORTED_QUERY",
                "intent": "unsupported",
                "confidence": confidence,
                "triggered_rules": triggered_rules,
                "taxonomy": "Outside Supported Clinical Scope",
                "ontology_links": [],
                "response": (
                    "Sorry, I cannot assist with this request because it falls outside my supported scope "
                    "(such as performing surgical procedures at home, veterinary medicine, controlled substances, "
                    "or non-healthcare topics). I can help with symptoms, medications, appointments, lab tests, and wellness."
                ),
            }

        # RULE 3: Low-Confidence Handling
        if confidence < self.confidence_threshold:
            triggered_rules.append(f"Rule 3: Low-Confidence Fallback ({confidence:.2f} < {self.confidence_threshold})")
            return {
                "status": "LOW_CONFIDENCE",
                "intent": predicted_intent,
                "confidence": confidence,
                "triggered_rules": triggered_rules,
                "taxonomy": "Ambiguous Query - Clarification Needed",
                "ontology_links": ["(Patient --[exhibits]--> Symptom)"],
                "response": (
                    "Your query is a bit brief or ambiguous, so I want to be sure before advising. "
                    "Could you clarify whether you are asking about symptoms, a medication, booking a doctor appointment, "
                    "lab test preparation, or preventive wellness?"
                ),
            }

        # Knowledge Retrieval for confident in-scope intent
        kb_entry = self.kb[predicted_intent]
        response_parts = [kb_entry["default_response"]]

        for kw, snippet in kb_entry["topic_snippets"].items():
            if kw in q_lower or kw in tokens:
                response_parts.append(snippet)

        # RULE 4: Medication & Prescription Safety Guardrail
        if predicted_intent == "medication_guidance":
            triggered_rules.append("Rule 4: Medication & Prescription Safety Guardrail")
            response_parts.append(
                "Safety Advisory: Always consult a licensed doctor or pharmacist before starting, stopping, or changing any medication."
            )

        # RULE 5: Persistent / Severe Symptom Clinical Triage
        if predicted_intent == "symptom_inquiry" and any(
            w in q_lower for w in ["severe", "days", "week", "persistent", "high fever", "sharp"]
        ):
            triggered_rules.append("Rule 5: Persistent/Severe Symptom Clinical Triage")
            response_parts.append(
                "Triage Note: Since you noted severe or ongoing symptoms, please schedule an in-person physician evaluation."
            )

        # RULE 6: Fasting & Pre-Test Preparation Advisory
        if predicted_intent == "diagnostic_lab_info" and any(
            w in q_lower for w in ["lipid", "cholesterol", "glucose", "sugar", "fast", "fasting", "ultrasound"]
        ):
            triggered_rules.append("Rule 6: Diagnostic Fasting & Prep Advisory")
            response_parts.append(
                "Preparation Reminder: Fasting blood tests and abdominal ultrasounds typically require 8–12 hours of fasting beforehand."
            )

        if not triggered_rules:
            triggered_rules.append("Standard Knowledge Retrieval Rule")

        return {
            "status": "SUCCESS",
            "intent": predicted_intent,
            "confidence": confidence,
            "triggered_rules": triggered_rules,
            "taxonomy": kb_entry["taxonomy_path"],
            "ontology_links": self._get_ontology_triples(kb_entry["concepts"]),
            "response": " ".join(response_parts),
        }

    def respond(self, user_query: str) -> dict:
        processed = preprocess_text(user_query)
        vec = self.vectorizer.transform([processed])
        probs = self.classifier.predict_proba(vec)[0]
        best_idx = int(np.argmax(probs))
        predicted_intent = self.classifier.classes_[best_idx]
        confidence = float(probs[best_idx])

        result = self._reason_and_retrieve(user_query, processed, predicted_intent, confidence)
        result["raw_query"] = user_query
        result["processed_query"] = processed
        return result


assistant = HealthcareCognitiveAssistant(
    vectorizer=vectorizer,
    classifier=classifier,
    knowledge_base=KNOWLEDGE_BASE,
    taxonomy=HEALTHCARE_TAXONOMY,
    concepts=DOMAIN_CONCEPTS,
    relationships=DOMAIN_RELATIONSHIPS,
)

# ==============================================================================
# AUTOMATED VERIFICATION OF ALL 5 REQUIRED TEST CASES
# ==============================================================================
test_cases = [
    ("Test 1A [Intent: symptom_inquiry]", "I have a sore throat and mild fever today"),
    ("Test 1B [Intent: medication_guidance]", "What are the side effects of paracetamol?"),
    ("Test 1C [Intent: appointment_booking]", "I would like to book an appointment with a cardiologist"),
    ("Test 1D [Intent: preventive_wellness]", "What daily exercise and diet keep the heart healthy?"),
    ("Test 1E [Intent: diagnostic_lab_info]", "What does a CBC blood test measure?"),
    ("Test 1F [Intent: emergency_alert]", "Someone collapsed and is unconscious on the floor"),
    ("Test 2  [Differently worded query]", "My throat feels scratchy and I have a bad headache"),
    ("Test 3  [Unsupported healthcare query]", "How can I perform surgical stitches at home on a deep wound?"),
    ("Test 4  [Low-confidence query]", "feeling weird check please"),
    ("Test 5  [Emergency-related query]", "I have crushing chest pain radiating to my left arm and difficulty breathing"),
]

print("\n" + "=" * 72)
print("AUTOMATED TEST SUITE RESULTS (ALL 5 SCENARIOS)")
print("=" * 72)
for label, query in test_cases:
    res = assistant.respond(query)
    print(f"\n{label}")
    print(f"  User Query      : {res['raw_query']}")
    print(f"  Preprocessed    : {res['processed_query']}")
    print(f"  Status / Intent : {res['status']} | {res['intent']} (Confidence: {res['confidence']:.2f})")
    print(f"  Taxonomy Path   : {res['taxonomy']}")
    print(f"  Ontology Links  : {', '.join(res['ontology_links']) if res['ontology_links'] else 'None'}")
    print(f"  Rules Triggered : {', '.join(res['triggered_rules'])}")
    print(f"  Assistant Reply : {res['response']}")

# ==============================================================================
# REQUIREMENT 10: GOOGLE COLAB INTERACTIVE INTERFACE (MULTIPLE QUERIES)
# ==============================================================================
def run_interactive_assistant():
    print("\n" + "=" * 72)
    print("HEALTHCARE COGNITIVE ASSISTANT - INTERACTIVE SESSION")
    print("Enter your healthcare queries below (type 'exit' or 'quit' to stop).")
    print("=" * 72)
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "stop"}:
            print("Assistant: Session ended. Stay healthy!")
            break
        output = assistant.respond(user_input)
        print(f"Assistant [{output['intent']} | Conf: {output['confidence']:.2f} | {output['status']}]:")
        print(f"  -> {output['response']}")
        print(f"  -> Taxonomy: {output['taxonomy']}")
        print(f"  -> Rules Applied: {', '.join(output['triggered_rules'])}")

# Launches the live multi-query interactive loop automatically in Google Colab
run_interactive_assistant()