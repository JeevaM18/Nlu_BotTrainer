from src.components.json_loader import IntentJSONLoader
from src.components.json_to_dataframe import flatten_intents_json
from src.components.evaluator import NLUEvaluator
from src.components.gemma_nlu import GemmaNLU
from src.utils.config_loader import load_config

if __name__ == "__main__":
    config = load_config()

    intents_path = config["paths"]["intents_path"]
    model_name = config["llm"]["model_name"]

    # Load intents.json
    loader = IntentJSONLoader(intents_path)
    intents_data = loader.load()

    # Flatten JSON → DataFrame
    df = flatten_intents_json(intents_data)

    # 🔥 IMPORTANT: sample while debugging
    df = df.sample(n=25, random_state=42).reset_index(drop=True)

    # Initialize Gemma
    nlu = GemmaNLU(model_name)

    # 🔥 ADD THIS BLOCK HERE
    predictions = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing: {row['text']}")
        result = nlu.predict(row["text"], intents_data)
        predictions.append(result["intent"])

    df["predicted_intent"] = predictions

    # Evaluate
    evaluator = NLUEvaluator()
    evaluator.evaluate(df)
