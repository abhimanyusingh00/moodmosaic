import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Make sure the project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.annotate import MoodMosaicPipeline

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MoodMosaic Dashboard",
    page_icon="🎭",
    layout="wide",
)

# ---------------------------------------------------------------------
# Global UI styling
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>

/* Dark background everywhere */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #050816;
    color: #f5f5f5;
}

/* Remove default white header band */
[data-testid="stHeader"] {
    background: transparent;
}

/* Remove top decorative bar */
[data-testid="stDecoration"] {
    background: transparent;
}

/* Tighter top padding */
div.block-container {
    padding-top: 1.5rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #050816;
    color: #f5f5f5;
}

/* Links */
a {
    color: #93c5fd;
}

/* Analyze button in red with hover effect */
div.stButton > button {
    background-color: #ff4b4b;
    color: #ffffff;
    border-radius: 999px;
    padding: 0.45rem 1.6rem;
    border: none;
    font-weight: 600;
    font-size: 0.95rem;
}

div.stButton > button:hover {
    background-color: #ff6b6b;
    color: #ffffff;
}

/* Tabs accent */
[data-baseweb="tab"] button[aria-selected="true"] {
    color: #ff6b6b;
    border-bottom: 2px solid #ff6b6b;
}

/* Data table header */
[data-testid="stTable"] th {
    background-color: #111827;
    color: #f5f5f5;
}

/* Labels */
label {
    color: #f5f5f5;
}

/* Radio and select text */
span[data-baseweb="typo"] {
    color: #f5f5f5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Example texts
# ---------------------------------------------------------------------
SAMPLES: Dict[str, str] = {
    "Positive day": """I am really excited about how this semester is going.
I feel proud of the work I put into my projects.
Thank you so much for helping me with the assignment.
I am grateful for my friends checking in on me today.
I am curious to see how far I can push this project.""",

    "Frustrated and stressed": """Today was exhausting and nothing worked the way I expected.
I am annoyed that my code keeps breaking right before the deadline.
This whole situation makes me nervous and a little scared.
I feel disappointed in myself for not planning better.
Please just fix this issue, I do not have the energy to debug again.""",

    "Polite vs impolite mix": """Could you please review this pull request when you have a moment?
This is wrong, fix it now.
I really appreciate the detailed feedback you gave me last week.
You never listen to instructions and it is getting frustrating.
Thank you again for taking the time to help.""",

    "Big Five flavored sample": """I love learning new ideas and exploring different research topics.
I enjoy reading and trying creative approaches in my projects.
I always plan my work carefully and double check everything.
I feel energized when I am around people and like starting conversations.
I often worry about results and deadlines when things are uncertain.""",
}

# ---------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MoodMosaic models...")
def load_pipeline() -> MoodMosaicPipeline:
    base_dir = ROOT_DIR

    # Prefer src/experiments/checkpoints if it exists, else experiments/checkpoints
    ckpt_dir = os.path.join(base_dir, "src", "experiments", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = os.path.join(base_dir, "experiments", "checkpoints")

    paths = {
        "emotion": os.path.join(ckpt_dir, "emotion", "best"),
        # IMPORTANT: directory only; annotate.py appends best_stanford.bin
        "tone": os.path.join(ckpt_dir, "tone_stanford"),
        "personality_ckpt": os.path.join(
            ckpt_dir, "personality_big5", "best_big5.pt"
        ),
        "personality_sbert": "sentence-transformers/all-mpnet-base-v2",
    }
    pipe = MoodMosaicPipeline(paths)
    return pipe


pipe: MoodMosaicPipeline = load_pipeline()

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _pick_first(result: Dict[str, Any], keys: List[str]) -> Any:
    """Return first non-None value for given keys without using `or` on arrays."""
    for k in keys:
        if k in result:
            return result[k]
    return None


def run_pipeline(texts: List[str]) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Call pipeline.annotate and normalize output shapes."""
    result: Dict[str, Any] = pipe.annotate(texts)

    # ---------- Emotion aggregate ----------
    emo_candidate = _pick_first(
        result,
        ["emotion_agg", "emo_agg", "emotions_agg", "emotion_vector"],
    )

    emo_agg: Dict[str, float] = {}

    if isinstance(emo_candidate, dict):
        emo_agg = {k: float(v) for k, v in emo_candidate.items()}
    elif emo_candidate is not None:
        try:
            arr = np.asarray(emo_candidate).ravel().tolist()
            labels = result.get("emotion_labels") or result.get("labels") or []
            if labels and len(labels) == len(arr):
                emo_agg = {lab: float(val) for lab, val in zip(labels, arr)}
            else:
                emo_agg = {f"emo_{i}": float(v) for i, v in enumerate(arr)}
        except Exception:
            emo_agg = {}

    # ---------- Personality ----------
    big5_candidate = _pick_first(result, ["personality", "big5", "big_five"])
    big5: Dict[str, float] = {}

    if isinstance(big5_candidate, dict):
        big5 = {k: float(v) for k, v in big5_candidate.items()}
    elif big5_candidate is not None:
        try:
            arr = np.asarray(big5_candidate).ravel().tolist()
            traits = ["O", "C", "E", "A", "N"]
            if len(arr) == 5:
                big5 = {traits[i]: float(arr[i]) for i in range(5)}
            else:
                big5 = {f"T{i}": float(v) for i, v in enumerate(arr)}
        except Exception:
            big5 = {}

    # ---------- Per message ----------
    per_df = result.get("per_message") or result.get("per_message_df") or []

    if isinstance(per_df, list):
        if per_df and isinstance(per_df[0], dict):
            per_df = pd.DataFrame(per_df)
        else:
            per_df = pd.DataFrame({"Text": texts})
    elif isinstance(per_df, pd.DataFrame):
        pass
    else:
        per_df = pd.DataFrame({"Text": texts})

    # Normalize column names
    col_map = {}
    for col in per_df.columns:
        low = col.lower()
        if low.startswith("text"):
            col_map[col] = "Text"
        elif "dominant" in low and "emotion" in low:
            col_map[col] = "Dominant emotion"
        elif "tone" in low or "polite" in low:
            col_map[col] = "Tone"
    per_df = per_df.rename(columns=col_map)

    return emo_agg, big5, per_df


def make_emotion_radar(emo_agg: Dict[str, float]) -> go.Figure:
    if not emo_agg:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            title="Not enough emotion signal yet.",
        )
        return fig

    labels = sorted(emo_agg.keys())
    values = [float(emo_agg.get(lab, 0.0)) for lab in labels]

    # Close loop
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="Emotions",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#050816",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_personality_radar(big5: Dict[str, float]) -> go.Figure:
    if not big5:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            title="No personality signal.",
        )
        return fig

    order = ["O", "C", "E", "A", "N"]
    traits = []
    values = []

    for key in big5.keys():
        letter = key[0].upper()
        if letter in order and letter not in traits:
            traits.append(letter)
            values.append(float(big5[key]))

    if not traits:
        traits = list(big5.keys())
        values = [float(v) for v in big5.values()]

    arr = np.array(values, dtype=float)
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = (arr + 1.0) / 2.0
    values = arr.tolist()

    traits.append(traits[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=traits,
            fill="toself",
            name="Personality",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#050816",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def top_emotions_summary(emo_agg: Dict[str, float], k: int = 3, min_score: float = 0.2) -> str:
    if not emo_agg:
        return "not enough signal yet."

    items = sorted(emo_agg.items(), key=lambda x: x[1], reverse=True)
    filtered = [lab for lab, score in items if score >= min_score]
    if not filtered:
        filtered = [lab for lab, _ in items[:k]]

    return ", ".join(filtered[:k])


def personality_summary(big5: Dict[str, float], k: int = 2) -> str:
    if not big5:
        return "none."

    items = sorted(big5.items(), key=lambda x: x[1], reverse=True)
    letters = [key[0].upper() for key, _ in items[:k]]
    return ", ".join(letters)


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    # Sidebar
    with st.sidebar:
        st.markdown("### MoodMosaic")
        st.write(
            "Explore how language reflects emotions, politeness tone, "
            "and a simple Big Five personality sketch. Paste text or upload a CSV."
        )

        input_mode = st.radio(
            "Input mode",
            ["Text box", "CSV upload"],
            index=0,
        )

        st.markdown("---")
        sample_name = st.selectbox(
            "Load an example",
            list(SAMPLES.keys()),
            index=0,
        )

        st.markdown("---")
        st.markdown(
            "Models: RoBERTa (emotion), DistilBERT (tone), "
            "SBERT plus MLP (personality)."
        )

    st.markdown("## MoodMosaic Dashboard")

    col_input, col_info = st.columns([3, 2])

    # Input column
    with col_input:
        st.markdown("### Input text")

        if input_mode == "Text box":
            default_text = SAMPLES[sample_name]
            raw = st.text_area(
                "",
                value=default_text,
                height=220,
                help="Enter one message per line.",
            )
            texts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            texts: List[str] = []
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                text_col = "text"
                if text_col not in df.columns:
                    text_col = df.columns[0]
                texts = [str(t).strip() for t in df[text_col].tolist() if str(t).strip()]
            raw = "\n".join(texts)

        analyze_clicked = st.button("Analyze")

    # Session info column
    with col_info:
        st.markdown("### Session info")
        device = "cpu"
        st.write(f"Device: **{device}**")
        st.write("Number of emotion labels: **28**")
        st.write("Tone classes: **impolite, neutral, polite**")
        st.write("Personality traits: **O, C, E, A, N**")
        if input_mode == "Text box":
            st.write(f"Messages detected: **{len(texts)}**")
        else:
            st.write(f"Messages from CSV: **{len(texts)}**")

    st.markdown("---")

    if analyze_clicked:
        if not texts:
            st.warning("Please enter at least one message or upload a CSV.")
            return

        with st.spinner("Running MoodMosaic pipeline..."):
            emo_agg, big5, per_df = run_pipeline(texts)

        st.markdown("### Summary")

        emo_summary = top_emotions_summary(emo_agg, k=3, min_score=0.2)
        st.write(
            "Top emotions across all messages: "
            f"**{emo_summary}**."
        )

        pers_summary = personality_summary(big5, k=2)
        st.write(
            "Strongest personality traits approximate: "
            f"**{pers_summary}**."
        )

        st.markdown("")

        tab_overview, tab_per = st.tabs(["Overview", "Per message details"])

        with tab_overview:
            col_emo, col_pers = st.columns(2)

            with col_emo:
                st.markdown("#### Emotion mosaic")
                emo_fig = make_emotion_radar(emo_agg)
                st.plotly_chart(emo_fig, use_container_width=True)

            with col_pers:
                st.markdown("#### Personality profile")
                pers_fig = make_personality_radar(big5)
                st.plotly_chart(pers_fig, use_container_width=True)

        with tab_per:
            st.markdown("### Per message analysis")

            if "Text" not in per_df.columns:
                per_df["Text"] = texts
            if "Dominant emotion" not in per_df.columns:
                per_df["Dominant emotion"] = ""
            if "Tone" not in per_df.columns:
                per_df["Tone"] = ""

            tone_options = sorted(
                [t for t in per_df["Tone"].unique().tolist() if t]
            )
            emo_options = sorted(
                [
                    e
                    for e in per_df["Dominant emotion"]
                    .unique()
                    .tolist()
                    if e
                ]
            )

            c1, c2 = st.columns(2)

            with c1:
                st.caption("Filter by tone")
                selected_tones = st.multiselect(
                    "",
                    options=tone_options,
                    default=tone_options,
                )

            with c2:
                st.caption("Filter by dominant emotion")
                selected_emos = st.multiselect(
                    "",
                    options=emo_options,
                    default=emo_options,
                )

            filtered = per_df.copy()
            if selected_tones:
                filtered = filtered[filtered["Tone"].isin(selected_tones)]
            if selected_emos:
                filtered = filtered[
                    filtered["Dominant emotion"].isin(selected_emos)
                ]

            st.dataframe(
                filtered[["Text", "Dominant emotion", "Tone"]],
                use_container_width=True,
                height=320,
            )

            with st.expander("Show full table without filters"):
                st.dataframe(
                    per_df[["Text", "Dominant emotion", "Tone"]],
                    use_container_width=True,
                    height=320,
                )


if __name__ == "__main__":
    main()
