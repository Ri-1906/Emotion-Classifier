# Lexicon — Text Emotion Detection

A small web app that reads a sentence and predicts the emotion behind it, using a Bidirectional GRU model trained on six emotion categories: sadness, joy, love, anger, fear, and surprise.

The backend is a FastAPI service that loads the model once at startup and exposes a prediction endpoint. The frontend is a single static HTML page served directly by FastAPI, with no build step or frontend framework involved.

<img width="950" height="499" alt="Screenshot 2026-09-23 152931" src="https://github.com/user-attachments/assets/6e3db6ed-9d0b-47a4-9d81-73c9f9e52766" />

<img width="944" height="497" alt="Screenshot 2026-09-23 153033" src="https://github.com/user-attachments/assets/29889139-0dd2-4e60-854a-5f2ca26cc0d5" />

<img width="947" height="487" alt="Screenshot 2026-09-23 153116" src="https://github.com/user-attachments/assets/4eaf4d9d-236a-479d-a6dc-4fc16ef92da8" />



## How it works

1. The user types a sentence into the page.
2. The frontend sends it to `POST /predict`.
3. The backend lowercases and strips the text, tokenizes it with the saved tokenizer, pads it to a fixed length, and runs it through the BiGRU model.
4. The response includes the top predicted emotion, a confidence score, and the full probability breakdown across all six emotions, which the page renders as an animated spectrum.

## Project structure

```
.
├── app.py                        # FastAPI app, model loading, endpoints
├── requirements.txt
├── runtime.txt                   # Python version pin for Render
├── Artifacts/
│   ├── BiGRU_Model_fixed.keras   # trained model
│   └── tokenizer.pkl             # fitted Keras tokenizer
└── static/
    └── index.html                # frontend (HTML, CSS, JS in one file)
```



## API endpoints

**`GET /`**
Serves the frontend page.

**`GET /health`**
Returns whether the model and tokenizer are loaded.

```json
{ "status": "Server is running", "model_loaded": true }
```

**`POST /predict`**
Takes a sentence and returns the predicted emotion.

Request:
```json
{ "text": "I can't believe this actually worked out." }
```

Response:
```json
{
  "text": "I can't believe this actually worked out.",
  "predicted_emotion": "surprise",
  "confidence": 0.81,
  "all_prob": {
    "sadness": 0.02,
    "joy": 0.11,
    "love": 0.01,
    "anger": 0.02,
    "fear": 0.03,
    "surprise": 0.81
  }
}
```

Sentences must be between 1 and 2000 characters. If the model hasn't finished loading yet, the endpoint returns a `503`.

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The app will be available at `https://emotion-classifier-twdw.onrender.com/`. Make sure `Artifacts/BiGRU_Model_fixed.keras` and `Artifacts/tokenizer.pkl` exist at the paths referenced in `app.py` before starting the server.

## Deploying on Render

1. Push the repo (including the `Artifacts/` folder) to GitHub.
2. Create a new Web Service on Render and point it at the repo.
3. Render will pick up `requirements.txt` and `runtime.txt` automatically.
4. Set the start command to:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Deploy. First boot will take a little longer while TensorFlow installs and the model loads — `GET /health` is a good way to confirm it's ready.

A couple of things worth knowing going in:
- The model files need to be small enough (or hosted somewhere fetchable) to fit inside Render's build — very large `.keras` files can make builds slow or fail on the free tier.
- `tensorflow-cpu` is used instead of the full `tensorflow` package to keep the build lighter, since inference here doesn't need GPU support.

## Notes on the model

- Input text is lowercased, stripped of punctuation and apostrophes, and collapsed to single spaces before tokenizing — matching the preprocessing used during training.
- Sequences are padded/truncated to a fixed length of 50 tokens.
- Emotion labels are fixed in this order: `sadness, joy, love, anger, fear, surprise`. If you retrain the model with a different label set or order, update `emotion_labels` and `emotion_emoji` in `app.py` and the `EMOJI` map in `static/index.html` to match.
