import streamlit as st
from src.components.json_loader import IntentJSONLoader
from src.components.gemma_nlu import GemmaNLU
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("STREAMLIT_APP")

st.set_page_config(page_title="BotTrainer", layout="centered")
st.title("🤖 BotTrainer – LLM-Based NLU")

config = load_config()
loader = IntentJSONLoader(config["paths"]["intents_path"])
intents_data = loader.load()

nlu = GemmaNLU(config["llm"]["model_name"])

user_text = st.text_input(
    "Enter a user message",
    placeholder="e.g., Book a flight to Delhi tomorrow"
)

if st.button("Predict Intent"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        logger.info(f"User input received: {user_text}")
        result = nlu.predict(user_text, intents_data)
        st.subheader("NLU Output")
        st.json(result)
