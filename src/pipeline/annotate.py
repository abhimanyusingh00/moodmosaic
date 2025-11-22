import os
import torch
import numpy as np
from transformers import AutoTokenizer
from src.models.emotion_model import EmotionClassifier
from src.models.tone_model import ToneClassifier
from src.models.personality_model import PersonalityRegressor
from src.pipeline.aggregation import summarize_emotions

# ----- Static labels so we do not depend on src.data at import time -----

GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

POLITE_LABELS = ["impolite", "neutral", "polite"]

BIG5_TRAITS = ["O", "C", "E", "A", "N"]


class MoodMosaicPipeline:
    def __init__(self, paths):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline using device:", self.device)

        # ----- Emotion model -----
        self.emo_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.emo_model = EmotionClassifier(
            model_name="roberta-base",
            num_labels=len(GOEMO_LABELS),
        )
        emo_ckpt = os.path.join(paths["emotion"], "pytorch_model.bin")
        if os.path.exists(emo_ckpt):
            emo_state = torch.load(emo_ckpt, map_location=self.device)
            self.emo_model.load_state_dict(emo_state)
            print("Loaded fine tuned emotion checkpoint")
        else:
            print("No emotion checkpoint found, using base RoBERTa with random head")
        self.emo_model.to(self.device)
        self.emo_model.eval()

        # ----- Tone model -----
        self.tone_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tone_model = ToneClassifier(
            model_name="distilbert-base-uncased",
            num_labels=len(POLITE_LABELS),
        )
        tone_ckpt = os.path.join(paths["tone"], "pytorch_model.bin")
        if os.path.exists(tone_ckpt):
            tone_state = torch.load(tone_ckpt, map_location=self.device)
            self.tone_model.load_state_dict(tone_state)
            print("Loaded fine tuned tone checkpoint")
        else:
            print("No tone checkpoint found, using base DistilBERT with random head")
        self.tone_model.to(self.device)
        self.tone_model.eval()

        # ----- Personality model -----
        self.personality_model = PersonalityRegressor(
            paths["personality_sbert"],
            hidden_dim=128,
        )
        pers_ckpt = paths["personality_ckpt"]
        if os.path.exists(pers_ckpt):
            pers_state = torch.load(pers_ckpt, map_location=self.device)
            self.personality_model.load_state_dict(pers_state)
            print("Loaded personality checkpoint")
        else:
            print("No personality checkpoint found, using random MLP weights")
        self.personality_model.to(self.device)
        self.personality_model.eval()

    @torch.no_grad()
    def analyze_messages(self, messages):
        emo_probs = []
        tones = []
        for text in messages:
            emo_probs.append(self._predict_emotion(text))
            tones.append(self._predict_tone(text))
        emo_probs = np.vstack(emo_probs)

        emo_summary = summarize_emotions(emo_probs, GOEMO_LABELS)
        joined_text = "\n".join(messages)
        pers = self._predict_personality(joined_text)

        return {
            "per_message": [
                {
                    "text": m,
                    "emotion_probs": emo_probs[i].tolist(),
                    "tone_label": tones[i],
                }
                for i, m in enumerate(messages)
            ],
            "summary": {
                "emotion": emo_summary,
                "personality": pers,
            },
        }

    def _predict_emotion(self, text):
        enc = self.emo_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.emo_model(**enc)
        logits = out["logits"].squeeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _predict_tone(self, text):
        enc = self.tone_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.tone_model(**enc)
        logits = out["logits"].squeeze(0)
        label_id = int(torch.argmax(logits).item())
        return label_id

    def _predict_personality(self, text):
        out = self.personality_model([text])
        preds = out["preds"].squeeze(0).cpu().numpy()
        return dict(zip(BIG5_TRAITS, preds.tolist()))
