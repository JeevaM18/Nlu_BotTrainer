import json

class IntentJSONLoader:
    def __init__(self, json_path):
        self.json_path = json_path

    def load(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "intents" in data, "❌ Missing 'intents' key"
        assert "entities" in data, "❌ Missing 'entities' key"

        for intent in data["intents"]:
            assert "name" in intent
            assert "examples" in intent
            assert "entities" in intent

        print("✅ intents.json loaded & validated")
        return data
