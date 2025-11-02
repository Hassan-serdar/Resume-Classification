import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import pickle
import re
import fitz 

app = FastAPI(title="Resume Classification API")

try:
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    clf = pickle.load(open("model/clf.pkl", "rb"))
    le = pickle.load(open("model/encoder.pkl", "rb"))
except FileNotFoundError:
    raise RuntimeError("Model files not found! Make sure clf.pkl, tfidf.pkl, and encoder.pkl are in the app/model/ directory.")

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

@app.post("/predict")
async def predict(resume_file: UploadFile = File(...)):

    if not resume_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        file_content = await resume_file.read()
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        
        pdf_document.close()

        cleaned_text = cleanResume(full_text)
        vectorized_text = tfidf.transform([cleaned_text])
        vectorized_text = vectorized_text.toarray()
        prediction_id = clf.predict(vectorized_text)[0]
        
        predicted_category = le.inverse_transform([prediction_id])[0]
        
        return {"filename": resume_file.filename, "predicted_category": predicted_category}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)