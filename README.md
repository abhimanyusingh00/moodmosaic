# MoodMosaic: Mapping Emotions and Personality Through Language

MoodMosaic is a small research prototype that shows how everyday language can reveal patterns of **emotion**, **politeness tone**, and **personality** in a simple, transparent way.

Given free text or a small CSV of messages, MoodMosaic:

- predicts **fine grained emotions** from the GoEmotions label set  
- predicts **politeness tone** (polite or impolite) from the Stanford Politeness Corpus  
- predicts a **Big Five personality profile** (O C E A N) using the Essays Big Five dataset  
- visualizes everything in an interactive **Streamlit dashboard** with radar charts and tables  

The goal is not clinical-level prediction, but a clear analytic view that feels like a mirror for the text you feed in.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Main Features](#MainFeatures)
3. [Data and Tasks](#data-and-tasks)  
4. [Screenshots and Visual Overview](#screenshots-and-visual-overview)  
5. [Models and Pipeline](#models-and-pipeline)    
6. [Installation](#installation)  
7. [Data and Checkpoints Setup](#data-and-checkpoints-setup)  
8. [Training the Models](#training-the-models)  
9. [Running the Dashboard](#running-the-dashboard)  
10. [Using the Dashboard](#using-the-dashboard)  
11. [Experiments and Results](#experiments-and-results)  
12. [Limitations and Ethics](#limitations-and-ethics)  
13. [Future Work](#future-work)  
---

## Project Overview

MoodMosaic combines three neural components into a single pipeline:

1. **Emotion classifier**  
   Fine-tunes a RoBERTa encoder on the **GoEmotions** dataset to predict multiple emotion labels per message.

2. **Tone classifier (politeness)**  
   Fine-tunes a DistilBERT encoder on the **Stanford Politeness** data (Cleanlab release) to predict **polite** vs **impolite** tone.

3. **Personality regressor (Big Five)**  
   Uses a **Sentence Transformers** encoder (e.g. `all-mpnet-base-v2`) plus a small feed-forward network to predict O C E A N scores from the **Essays Big Five** corpus.

An aggregation layer combines per-message outputs into:

- a global **emotion distribution**  
- a **Big Five personality profile** for the text in the current session  
- a **per-message table** with text, dominant emotion, and tone  

All results are presented in a simple **Streamlit app**.

---

## Main Features

- **Two input modes**
  - Free text input, with one message per line  
  - CSV upload, where you can choose which column contains text  

- **Three prediction heads**
  - Emotion (multi-label, GoEmotions)  
  - Tone (binary politeness)  
  - Personality (five continuous traits: O, C, E, A, N)  

- **Interactive dashboard**
  - Emotion radar chart  
  - Personality radar chart  
  - Per-message table with filters  
  - Downloadable CSV of predictions  

- **Reproducible experiments**
  - Training scripts for all three models  
  - Checkpoints saved under `experiments/checkpoints/...`  
  - Experiment configs under `configs/`  

---
## Data and Tasks
Data and Tasks
MoodMosaic uses three publicly available datasets (not redistributed in this repository).
You are expected to download them yourself from their official sources or common ML hubs.
1. Emotion: GoEmotions
Task: Multi-label emotion classification over short comments.
Labels: 27 emotion labels + neutral (subset of the full GoEmotions label space).
Use in MoodMosaic:
Each message may express multiple emotions. The model outputs a probability per label; a threshold is applied at inference time to decide which labels are “active.”
Example labels (non-exhaustive): joy, anger, sadness, gratitude, fear, disappointment, admiration, confusion.
2. Tone: Politeness
Task: Binary classification: impolite (0) vs polite (1).
Data: Short requests and responses annotated for politeness.
Use in MoodMosaic:
Each message gets a single politeness label. The dashboard overlays this with emotions to show how emotional tone and politeness co-occur in the text.
3. Personality: Essays Big Five
Task: Regression of Big Five traits from essays.
Traits:
Openness
Conscientiousness
Extraversion
Agreeableness
Neuroticism
Use in MoodMosaic:
Text is aggregated at the session level (or batched) and fed through a personality regressor, producing continuous scores for each trait. These are visualized in a radar chart.

## Limitations and Ethics
MoodMosaic is a research and teaching prototype, not a psychological assessment or production system.
Key limitations:
Data bias and coverage
The training data comes from specific platforms and contexts.
The models may not generalize to all languages, dialects, or genres.
No clinical validity
The Big Five scores are rough, text-based approximations.
They are not validated psychological measurements.
Over-interpretation risk
Radar charts and tables can look very precise.
Users should treat them as exploratory visualizations, not ground truth.
Privacy and safety
Do not feed in sensitive, private, or confidential data.
If you use real chat logs or emails, anonymize them first.
Small politeness dataset
The politeness component is trained on a relatively small slice of data.
High validation accuracy may not carry over to all real-world requests.
Because of these limitations, do not use MoodMosaic for:
hiring or performance evaluation
medical or mental health decisions
grading or disciplinary decisions
any other high-stakes setting

## Using the Dashboard
Input modes
Free text mode
Paste or type messages into the text area.
Use one message per line (e.g., chat turns, comments, tweets).
Click the button to run the analysis.
CSV upload mode
Upload a .csv file (e.g., team_chat.csv).
Choose which column contains the text.
Optionally filter or sample rows.
Output views
Overview tab
Emotion radar chart: shows the average emotion distribution across all messages.
Big Five radar chart: shows the aggregated OCEAN profile for the current session.
Per-message details tab
Table with one row per message:
original text
dominant emotion label
politeness label (polite/impolite)
Filters to subset by emotion or tone.
A button to download predictions as CSV.
Preset examples
The app can optionally offer presets such as:
An artificially positive conversation.
A more frustrated or negative conversation.
Mixed, realistic chat logs.
Synthetic examples highlighting particular traits.
These help you demo the system quickly without preparing your own data.


## Future Work

Possible extensions and improvements:
Better calibration and uncertainty estimates
Improve how emotion and politeness probabilities are calibrated and communicated.
Richer conversation analysis
Per-speaker emotion and personality views.
Time-series plots over long chats.
Clustering of messages by emotional style.
Multilingual support
Add language identification and fine-tuned models for non-English text.
Explainability tools
Token-level or span-level saliency maps.
Natural language explanations of why a message was labeled a certain way.
Larger and more diverse datasets
Stronger politeness models with more data.
Additional personality datasets beyond essays.
Contributions via issues and pull requests are welcome.

## 

## Screenshots and Visual Overview

### 1. Overall system pipeline

High-level architecture: text input → preprocessing → per-message models → aggregation → personality → dashboard.

```markdown
![System pipeline diagram](pipeline_overview.png)
![Dashboard overview with emotion + Big Five radar charts](dashboard_overview.png)
![Per-message details table](dashboard_message_table.png)
![Emotion radar chart over GoEmotions labels](emotion_radar.png)
![Big Five personality radar chart](personality_radar.png)


