import torch
import numpy as np
from transformers import AutoTokenizer
from src.models.emotion_model import EmotionClassifier
from src.models.tone_model import ToneClassifier
from src.models.personality_model import PersonalityRegressor
from src.data.goemotions import GOEMO_LABELS
from src.data.politeness import POLITE_LABELS
from src.data.essays_big5 import BIG5_TRAITS
from src.pipeline.aggregation import summarize_emotions

class MoodMosaicPipeline:
    def __init__(self, paths):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline using device:", self.device)

        # ----- Emotion (RoBERTa base + our checkpoint) -----
        self.emo_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.emo_model = EmotionClassifier(
            model_name="roberta-base",
            num_labels=len(GOEMO_LABELS),
        )
        emo_state = torch.load(
            f'{paths["emotion"]}/pytorch_model.bin',
            map_location=self.device,
        )
        self.emo_model.load_state_dict(emo_state)
        self.emo_model.to(self.device)
        self.emo_model.eval()

        # ----- Tone (DistilBERT base + our checkpoint) -----
        self.tone_tokenizer = AutoTokenizer.from_pretrained(paths["tone"])
        self.tone_model = ToneClassifier(
            model_name="distilbert-base-uncased",
            num_labels=len(POLITE_LABELS),
        )
        tone_state = torch.load(
            f'{paths["tone"]}/pytorch_model.bin',
            map_location=self.device,
        )
        self.tone_model.load_state_dict(tone_state)
        self.tone_model.to(self.device)
        self.tone_model.eval()

        # ----- Personality (SentenceTransformer + our MLP, hidden_dim=128) -----
        self.personality_model = PersonalityRegressor(
            paths["personality_sbert"],
            hidden_dim=128,
        )
        pers_state = torch.load(
            paths["personality_ckpt"],
            map_location=self.device,
        )
        self.personality_model.load_state_dict(pers_state)
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
