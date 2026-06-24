# src/medical_terms.py

MEDICAL_TERMS = {
    "chemotherapy": "A treatment that uses medicines to destroy cancer cells.",
    
    "radiation therapy": "A treatment that uses high-energy rays to kill cancer cells.",
    
    "stage ii breast cancer": "Cancer is present in the breast and has spread slightly to nearby tissues or lymph nodes.",
    
    "tumor": "An abnormal growth of cells in the body.",
    
    "malignant": "Cancerous and able to spread to other parts of the body.",
    
    "benign": "Not cancerous and does not spread."
}

def simplify_terms(text):
    explanations = []

    text_lower = text.lower()

    for term, meaning in MEDICAL_TERMS.items():
        if term in text_lower:
            explanations.append(
                f"{term.title()}:\n{meaning}\n"
            )

    return "\n".join(explanations)