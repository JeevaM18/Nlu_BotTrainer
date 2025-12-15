# 🤖 BotTrainer – LLM-Based NLU Model Trainer & Evaluator

BotTrainer is an end-to-end **LLM-based Natural Language Understanding (NLU)** system that performs **intent classification and entity extraction** using prompt engineering instead of traditional ML classifiers.  
The system uses a **JSON-first dataset**, a **local Gemma-3 model**, and provides **evaluation metrics and an interactive Streamlit UI**.

---

## 📌 Project Objectives

- Build an NLU pipeline using **Large Language Models (LLMs)**
- Replace classical intent classifiers with **prompt-based inference**
- Perform **intent detection and entity extraction**
- Evaluate performance using **Accuracy, Precision, Recall, and F1-score**
- Provide a **real-time chatbot-style interface**
- Follow **production-style project structure**

---

## 🧠 Key Features

- JSON-based intent & entity schema (`intents.json`)
- Prompt engineering with schema-guided constraints
- Local LLM inference using **Gemma-3 (via Ollama)**
- Automatic JSON parsing & validation
- Evaluation pipeline with confusion matrix
- Streamlit web interface for live testing

---

## 📦 Dataset Design

### Primary Dataset: `intents.json`
- Defines **intents**, **training examples**, and **entity schema**
- Used directly in prompt construction

Example:
```json
{
  "name": "book_flight",
  "examples": [
    "Book a flight to Delhi",
    "I want to go to Mumbai tomorrow"
  ],
  "entities": ["location", "date"]
}