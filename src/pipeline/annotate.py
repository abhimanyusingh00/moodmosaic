import os
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.personality_model import PersonalityRegressor

# ---------------------------
# Label definitions
# ---------------------------

GOEMO_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

POLITE_LABELS: List[str] = ["impolite", "polite"]

BIG5_TRAITS: List[str] = ["O", "C", "E", "A", "N"]


class MoodMosaicPipeline:
    """
    Wraps three components:

      - Emotion classifier (RoBERTa fine-tuned on GoEmotions)
      - Politeness classifier (DistilBERT fine-tuned on Stanford Politeness)
      - Personality regressor (SBERT + MLP trained on Essays Big Five)

    Dashboard calls:  result = pipe.annotate(texts)
    """

    def __init__(self, paths: Dict[str, str]) -> None:
        """
        Args:
            paths: dictionary with keys
                - 'emotion'           -> directory with emotion checkpoint
                - 'tone'              -> directory containing best_stanford.bin
                - 'personality_sbert' -> SBERT model name (e.g. all-mpnet-base-v2)
                - 'personality_ckpt'  -> path to best_big5.pt
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Emotion model (GoEmotions / RoBERTa) ----------------
        emo_dir = paths["emotion"]

        # Find a checkpoint file inside emo_dir
        emo_ckpt = None
        for fname in ["pytorch_model.bin", "best.bin", "best.pt", "model.bin"]:
            candidate = os.path.join(emo_dir, fname)
            if os.path.exists(candidate):
                emo_ckpt = candidate
                break

        if emo_ckpt is None:
            raise FileNotFoundError(
                f"Could not find emotion checkpoint in {emo_dir}. "
                "Expected one of: pytorch_model.bin, best.bin, best.pt, model.bin"
            )

        # Load base RoBERTa config and then load our fine-tuned weights
        self.emo_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.emo_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=len(GOEMO_LABELS),
        )
        emo_state = torch.load(emo_ckpt, map_location=self.device)

        # Training script might have wrapped this in a dict
        if isinstance(emo_state, dict) and "model_state_dict" in emo_state:
            emo_state = emo_state["model_state_dict"]

        self.emo_model.load_state_dict(emo_state, strict=False)
        self.emo_model.to(self.device)
        self.emo_model.eval()

        # ---------------- Tone model (DistilBERT politeness) ----------------
        tone_ckpt_dir = paths["tone"]
        tone_ckpt_path = os.path.join(tone_ckpt_dir, "best_stanford.bin")

        self.tone_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tone_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(POLITE_LABELS),
        )
        tone_state = torch.load(tone_ckpt_path, map_location=self.device)
        if isinstance(tone_state, dict) and "model_state_dict" in tone_state:
            tone_state = tone_state["model_state_dict"]
        self.tone_model.load_state_dict(tone_state, strict=False)
        self.tone_model.to(self.device)
        self.tone_model.eval()

        # ---------------- Personality model (SBERT + MLP) ----------------
        self.personality_model = PersonalityRegressor(
            sbert_model=paths["personality_sbert"],
            hidden_dim=128,
        )
        pers_state = torch.load(paths["personality_ckpt"], map_location=self.device)
        if isinstance(pers_state, dict) and "model_state_dict" in pers_state:
            pers_state = pers_state["model_state_dict"]
        self.personality_model.load_state_dict(pers_state, strict=False)
        self.personality_model.to(self.device)
        self.personality_model.eval()

    # =================== Internal helpers ===================

    def _predict_emotions(self, texts: List[str]) -> np.ndarray:
        """Return per-message emotion scores in [0,1], shape (N, num_labels)."""
        all_probs: List[np.ndarray] = []

        self.emo_model.eval()
        with torch.no_grad():
            for text in texts:
                enc = self.emo_tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.emo_model(**enc).logits  # [1, L]
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                all_probs.append(probs)

        return np.vstack(all_probs)

    def _predict_tone(self, texts: List[str]) -> List[str]:
        """Return per-message tone labels ('impolite' or 'polite')."""
        labels: List[str] = []

        self.tone_model.eval()
        with torch.no_grad():
            for text in texts:
                enc = self.tone_tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.tone_model(**enc).logits  # [1, 2]
                probs = torch.softmax(logits, dim=-1)[0]
                idx = int(torch.argmax(probs).item())
                labels.append(POLITE_LABELS[idx])

        return labels

    def _personality_to_vector(self, out: Any) -> np.ndarray:
        """
        Try to convert whatever PersonalityRegressor returned into a 1D numpy
        vector of length len(BIG5_TRAITS).
        """
        # Case 1: plain tensor [1,5] or [5]
        if isinstance(out, torch.Tensor):
            vec = out.squeeze(0).detach().cpu().numpy().astype(float)
            return vec

        # Case 2: dict with traits as keys
        if isinstance(out, dict):
            if all(k in out for k in BIG5_TRAITS):
                vals = [float(out[k]) for k in BIG5_TRAITS]
                return np.array(vals, dtype=float)
            if "scores" in out:
                scores = out["scores"]
                return self._personality_to_vector(scores)

        # Case 3: list or tuple
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                return np.zeros(len(BIG5_TRAITS), dtype=float)
            first = out[0]
            # maybe a tensor row
            if isinstance(first, torch.Tensor):
                return first.detach().cpu().numpy().astype(float)
            if isinstance(first, (list, tuple, np.ndarray)):
                arr = np.array(first, dtype=float)
                return arr
            if isinstance(first, dict):
                return self._personality_to_vector(first)

        # Fallback: try to coerce to array of numbers
        try:
            arr = np.array(out, dtype=float).reshape(-1)
            if arr.size >= len(BIG5_TRAITS):
                return arr[: len(BIG5_TRAITS)]
        except Exception:
            pass

        # Ultimate fallback: neutral 0 vector
        return np.zeros(len(BIG5_TRAITS), dtype=float)

    def _predict_personality(self, texts: List[str]) -> Dict[str, float]:
        """
        Collapse all messages into one block, run Big Five regressor, and
        normalize scores from roughly [-1, 1] into [0, 1].
        """
        if not texts:
            return {t: 0.5 for t in BIG5_TRAITS}

        joined = " ".join(texts)

        self.personality_model.eval()
        with torch.no_grad():
            out = self.personality_model([joined])

        raw_vec = self._personality_to_vector(out)

        # Map approx [-1, 1] to [0, 1]
        norm = (raw_vec + 1.0) / 2.0
        norm = np.clip(norm, 0.0, 1.0)

        return {trait: float(val) for trait, val in zip(BIG5_TRAITS, norm)}

    # =================== Public API ===================

    def annotate(self, texts: List[str]) -> Dict[str, Any]:
        """
        Main entry point used by the Streamlit app.

        Args:
            texts: list of messages, one per line.

        Returns:
            dict with keys:
              - 'emotion_labels'
              - 'emotion_agg'
              - 'personality'
              - 'per_message'
        """
        if not texts:
            return {
                "emotion_labels": GOEMO_LABELS,
                "emotion_agg": np.zeros(len(GOEMO_LABELS)),
                "personality": {t: 0.5 for t in BIG5_TRAITS},
                "per_message": [],
            }

        emo_scores = self._predict_emotions(texts)  # (N, L)
        emo_agg = emo_scores.mean(axis=0)

        dom_idx = emo_scores.argmax(axis=1)
        dom_emotions = [GOEMO_LABELS[i] for i in dom_idx]

        tones = self._predict_tone(texts)
        personality = self._predict_personality(texts)

        per_message = []
        for text, emo, tone in zip(texts, dom_emotions, tones):
            per_message.append(
                {
                    "text": text,
                    "dominant_emotion": emo,
                    "tone": tone,
                }
            )

        return {
            "emotion_labels": GOEMO_LABELS,
            "emotion_agg": emo_agg,
            "personality": personality,
            "per_message": per_message,
        }

    # Optional alias used elsewhere if needed
    def run(self, texts: List[str]) -> Dict[str, Any]:
        return self.annotate(texts)


