import json
import subprocess
import re
from src.utils.prompt_template import build_nlu_prompt
from src.utils.logger import setup_logger

logger = setup_logger("GEMMA_NLU")

class GemmaNLU:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, text, intents_data):
        logger.info(f"Running inference for text: {text}")

        prompt = build_nlu_prompt(text, intents_data)

        process = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout_bytes, _ = process.communicate(
            input=prompt.encode("utf-8", errors="ignore")
        )

        raw_output = stdout_bytes.decode("utf-8", errors="ignore").strip()

        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        if not json_match:
            logger.warning("No JSON found in model output")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {}
            }

        try:
            output = json.loads(json_match.group())
            return output
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {}
            }
