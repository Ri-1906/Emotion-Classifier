from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import re
from pydantic import BaseModel,Field
from keras.models import  load_model
import pickle
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequences import pad_sequences
import numpy as np
from keras.utils import pad_sequences 


from keras.layers import Embedding, Dense


class CompatibleEmbedding(Embedding):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)

    
# @app.get('/')
# def greet():
    # return {"Helloooo!!"}

#model path
model_path = "Artifacts/BiGRU_Model_fixed.keras"

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






# load the model and tokenizer once the server starts

dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer")
    dl_model['BiGRU'] = load_model(model_path,compile=False)
    # dl_model['BiGRU'] = load_model(
    #     model_path,
    #     custom_objects={
    #         "Embedding": CompatibleEmbedding,
    #         "Dense": CompatibleDense
    #     },
    #     compile=False
    # )
    with open(tokenizer_path, 'rb') as file:
        dl_model['tokenizer'] = pickle.load(file)
    print("Models loaded successfully")

    yield # Pause, model is loaded and server is running and at this point model is waiting

    dl_model.clear()





app = FastAPI(
    lifespan=lifespan
)

# @app.get('/')
# def greet():
#     return {"Helloooo!!"}



# Mount the static files to FastAPI app
# enable CORS to allow requests from diff origin


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



app.mount('/static', StaticFiles(directory = 'static'),name = 'static')


"""API Endpoints
1. Server UI at homepage ('/')
2. Health Check Endpoint ('/health')
3. Predict EMotion endpoint ('/predict')
"""


# 1.
@app.get('/', include_in_schema=False)
def serve_ui():
    return FileResponse('static/index.html')

# 2.
@app.get('/health',response_model=healthResponse)
def healthCheck():
    return healthResponse(status="Server is running",model_loaded=bool(dl_model))


# 3.
@app.post('/predict',response_model=predictResponse)
def predict(text_input: textInput):
    
    BiGRU_model = dl_model.get('BiGRU')
    tokenizer_model = dl_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    # cleanse the input sentences
    cleaned_text = preprocess_text(text_input.text)

    # convert words to tokens
    # pad the sequences 
    
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_len,
        padding='post',
        truncating='post',
    )


    # run prediction using BiGRU model

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_prob = {
        label : float(prob) for prob, label in zip(probabilities,emotion_labels)
    }

    return predictResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence =float(probabilities[top_emotion_index]),
        all_prob = all_prob
    )




    # return the top emotion with full prob breakdown
