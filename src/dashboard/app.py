import os
import sys

# --- Add project root to sys.path so "src" package can be imported ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.annotate import MoodMosaicPipeline
from src.data.goemotions import GOEMO_LABELS
from src.data.politeness import POLITE_LABELS

paths = {
    "emotion": "experiments/checkpoints/emotion/best",
    "tone": "experiments/checkpoints/tone/best",
    "personality_sbert": "sentence-transformers/all-mpnet-base-v2",
    "personality_ckpt": "experiments/checkpoints/personality/best.pt",
}

@st.cache_resource
def load_pipeline():
    return MoodMosaicPipeline(paths)

pipe = load_pipeline()

st.title("MoodMosaic")
st.write("Enter one message per line.")

user_input = st.text_area("Text", height=200)

if st.button("Analyze") and user_input.strip():
    messages = [m.strip() for m in user_input.split("\n") if m.strip()]
    result = pipe.analyze_messages(messages)

    # Emotion radar
    emo_summary = result["summary"]["emotion"]
    emo_df = pd.DataFrame(
        {"emotion": list(emo_summary.keys()), "score": list(emo_summary.values())}
    )
    fig_emo = px.line_polar(
        emo_df,
        r="score",
        theta="emotion",
        line_close=True,
    )
    st.subheader("Emotion mosaic")
    st.plotly_chart(fig_emo, use_container_width=True)

    # Personality radar
    pers = result["summary"]["personality"]
    pers_df = pd.DataFrame(
        {"trait": list(pers.keys()), "score": list(pers.values())}
    )
    fig_pers = px.line_polar(
        pers_df,
        r="score",
        theta="trait",
        line_close=True,
    )
    st.subheader("Personality profile")
    st.plotly_chart(fig_pers, use_container_width=True)

    # Per message table
    st.subheader("Per message breakdown")
    tone_map = {i: name for i, name in enumerate(POLITE_LABELS)}
    rows = []
    for row in result["per_message"]:
        probs = row["emotion_probs"]
        dominant_idx = int(max(enumerate(probs), key=lambda x: x[1])[0])
        dominant_emo = GOEMO_LABELS[dominant_idx]
        rows.append(
            {
                "Text": row["text"],
                "Dominant emotion": dominant_emo,
                "Tone": tone_map.get(row["tone_label"], "Unknown"),
            }
        )
    st.dataframe(pd.DataFrame(rows))
