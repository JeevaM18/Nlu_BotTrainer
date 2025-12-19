import streamlit as st
import pandas as pd

from src.components.json_loader import IntentJSONLoader
from src.components.json_to_dataframe import flatten_intents_json
from src.components.gemma_nlu import GemmaNLU
from src.components.evaluator import NLUEvaluator
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

# ----------------- SETUP -----------------
logger = setup_logger("STREAMLIT_APP")

st.set_page_config(
    page_title="BotTrainer – LLM NLU Platform",
    layout="wide",
    page_icon="🤖"
)

# ----------------- LOAD CONFIG & DATA -----------------
config = load_config()
loader = IntentJSONLoader(config["paths"]["intents_path"])
intents_data = loader.load()

df = flatten_intents_json(intents_data)   # <-- contains text, true_intent
nlu = GemmaNLU(config["llm"]["model_name"])
evaluator = NLUEvaluator()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🤖 BotTrainer")
    st.caption("LLM-Based NLU Platform")

    st.divider()

    page = st.radio(
        "📌 Navigation",
        ["NLU Tester", "Evaluation", "Dataset Overview"]
    )

    st.divider()

    st.markdown("### ⚙️ Model Info")
    st.write("**Model:** Gemma-3 (Local)")
    st.write("**Inference:** Ollama")
    st.write("**Total Intents:**", df["true_intent"].nunique())

    st.divider()
    st.caption("Built by Jeeva M • Portfolio Project")

# ----------------- NLU TESTER -----------------
if page == "NLU Tester":
    st.title("✨ NLU Tester")
    st.caption("Real-time intent classification and entity extraction")

    user_text = st.text_area(
        "Enter a user message",
        placeholder="e.g., Book a flight to Delhi tomorrow",
        height=120
    )

    if st.button("🚀 Analyze Message", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text")
        else:
            logger.info(f"User input: {user_text}")
            result = nlu.predict(user_text, intents_data)

            st.subheader("🔍 Prediction Result")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🏷 Intent")
                st.success(result["intent"])

            with col2:
                st.markdown("### 🎯 Confidence")
                st.progress(float(result.get("confidence", 0)))

            st.markdown("### 🧩 Extracted Entities")
            entities = result.get("entities", {})
            if entities:
                for k, v in entities.items():
                    st.info(f"**{k}** : {v}")
            else:
                st.write("No entities detected")

# ----------------- EVALUATION -----------------
elif page == "Evaluation":
    st.title("📊 Model Evaluation")
    st.caption("Evaluation on 20 intents (1 sample per intent)")

    if st.button("▶ Run Evaluation"):
        logger.info("Evaluation started")

        # 1 sample per intent → 20 intents total
        df_eval = (
            df.groupby("true_intent", group_keys=False)
              .head(1)
              .reset_index(drop=True)
        )

        predictions = []
        for text in df_eval["text"]:
            result = nlu.predict(text, intents_data)
            predictions.append(result["intent"])

        df_eval["predicted_intent"] = predictions

        metrics, cm = evaluator.evaluate_with_results(
            df_eval.rename(columns={"true_intent": "intent"})
        )

        st.subheader("📈 Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        c4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

        st.subheader("🔀 Confusion Matrix")
        st.dataframe(cm, use_container_width=True)

# ----------------- DATASET OVERVIEW -----------------
elif page == "Dataset Overview":
    st.title("📁 Dataset Overview")
    st.caption("Intent distribution and dataset structure")

    st.subheader("📌 Intent Distribution")
    st.bar_chart(df["true_intent"].value_counts())

    st.subheader("🔍 Sample Data")
    st.dataframe(df.sample(10), use_container_width=True)

    st.subheader("📊 Dataset Summary")
    st.write({
        "Total Samples": len(df),
        "Total Intents": df["true_intent"].nunique(),
        "Avg Samples per Intent": round(len(df) / df["true_intent"].nunique(), 2)
    })
