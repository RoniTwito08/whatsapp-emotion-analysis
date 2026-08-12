# WhatsApp Behavioral Analysis

**Early detection of interest shifts in WhatsApp conversations using NLP, behavioral signals, Transformers, and machine learning.**

This project explores whether changes in conversational interest can be detected **before they become obvious**, by combining the semantic content of messages with behavioral patterns such as response timing, message frequency, message length, and conversation dynamics.

The system compares multiple modeling approaches to evaluate whether behavioral information can improve predictions beyond text-only models.

## 🎯 Project Goal

The goal is to classify WhatsApp conversations into two behavioral states:

* **Interested**
* **Losing Interest**

Rather than relying only on sentiment or emotion, the project analyzes both **what people say** and **how their communication behavior changes over time**.

A major focus of the project is **early detection** — determining how soon an interest shift can be identified using only a partial conversation.

## 🧠 Approach

The project combines several complementary approaches:

### Transformer-Based NLP

Transformer models are fine-tuned on WhatsApp conversations to learn semantic patterns associated with changes in conversational interest.

### Behavioral Analysis

The system extracts behavioral signals from conversation structure, including:

* Response timing patterns
* Message frequency
* Message length
* Conversation initiation patterns
* Structural changes throughout the conversation

### Behavioral Baseline

A machine-learning baseline is trained using only behavioral features, allowing us to measure how much predictive information exists independently of message content.

### Text + Behavioral Fusion

Text representations from the Transformer model are combined with behavioral features to investigate whether multimodal conversational signals improve classification performance.

### Early Detection

The models are evaluated on partial conversations to determine how early they can identify a potential shift in interest.

## 🔬 Experiments

The project includes several experimental setups:

* Text-only Transformer classification
* Behavioral-only baseline
* Text + behavioral feature fusion
* Feature ablation experiments
* Continued fine-tuning experiments
* Early-detection evaluation at different conversation stages

These experiments allow us to compare the contribution of semantic and behavioral information and understand which signals are most useful for detecting interest shifts.

## 📊 Dataset

The dataset contains **3,055 labeled WhatsApp conversations**, balanced between the two target classes.

| Split      | Conversations |
| ---------- | ------------: |
| Training   |         2,137 |
| Validation |           459 |
| Test       |           459 |
| **Total**  |     **3,055** |

The dataset is split into approximately **70% training, 15% validation, and 15% testing**, while maintaining balanced class distributions.

## 🛠 Tech Stack

**Language**

Python

**Machine Learning & NLP**

* Transformers
* Deep Learning
* Scikit-learn
* Transformer fine-tuning
* Behavioral feature engineering

**Experimentation**

* Ablation studies
* Model fusion
* Early-detection evaluation
* Behavioral baselines

## 📁 Project Structure

```text
whatsapp-emotion-analysis/
│
├── text-model/
│   ├── behavioral_features_design.py
│   ├── behavioral_baseline.py
│   ├── pure_behavioral_baseline.py
│   ├── fusion_train.py
│   ├── ablation_continued_finetune.py
│   ├── early_detection_e1.py
│   ├── early_detection_e2.py
│   └── tests/
│
├── outputs/
│   └── split_ids.json
│
├── config.json
├── requirements.txt
└── README.md
```

## 🧪 Evaluation

The project evaluates models from several perspectives:

**Classification Performance**
How accurately can the model distinguish between interested and losing-interest conversations?

**Behavioral Contribution**
Do behavioral signals provide useful information beyond the text itself?

**Feature Importance**
Which behavioral signals contribute most to prediction performance?

**Early Detection**
How much of a conversation is required before the model can reliably identify an interest shift?

## 🎓 Research Motivation

Traditional conversation analysis often focuses primarily on sentiment or emotion.

However, conversational interest can also appear through changes in **behavior** — slower responses, shorter messages, reduced initiation, or changes in communication patterns.

This project investigates whether combining these behavioral signals with modern NLP models can provide a more complete representation of conversational dynamics.

## 👥 Authors

**Roni Twito**
M.Sc. Computer Science — Artificial Intelligence & Algorithms

**Matan Zohar Cohen**

---

*Developed as part of an NLP research project exploring behavioral and semantic signals in WhatsApp conversations.*
