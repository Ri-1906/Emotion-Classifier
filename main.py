from fastapi import FastAPI
import re
from pydantic import BaseModel,Field

app = FastAPI()

@app.get('/')
def greet():
    return {"Helloooo!!"}

#model path
model_path = "Artifacts/BiGRU_Model.keras"

#tokenizer path
tokenizer_path = "Artifacts/tokenizer.pkl"

#max sequence length
max_sequence_len = 50

#emotion labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

#emotion emojis
emotion_emoji = {
    'sadness' : '😢',
    'joy': '😀',
    'love' : '❤️', 
    'anger' : '😠', 
    'fear' : '😨', 
    'surprise': '😮'
}

# Clean raw text so it matches format used while training
# convert text to lowercase, remove special characters and punctuation and extra spaces
def preprocess_text(text : str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text



#  req and res schema

class textInput(BaseModel):
    text : str = Field(
            ...,
            min_length=1,
            max_length=2000,
            description="The sentence to analyze"
        )

class predictResponse(BaseModel):
    text : str
    predicted_emotion :str
    confidence : float
    all_prob : dict[str,float]


class healthResponse(BaseModel):
    status : str
    model_loaded : bool


