import json
from src.utils.logger import setup_logger

logger = setup_logger("JSON_LOADER")

class IntentJSONLoader:
    def __init__(self, json_path):
        self.json_path = json_path

    def load(self):
        logger.info(f"Loading intents file from {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "intents" in data, "Missing 'intents' key"
        assert "entities" in data, "Missing 'entities' key"

        for intent in data["intents"]:
            assert "name" in intent
            assert "examples" in intent
            assert "entities" in intent

        logger.info("intents.json loaded & validated successfully")
        return data
