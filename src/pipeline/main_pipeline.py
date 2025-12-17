from src.components.json_loader import IntentJSONLoader
from src.components.json_to_dataframe import flatten_intents_json
from src.components.evaluator import NLUEvaluator
from src.components.gemma_nlu import GemmaNLU
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("MAIN_PIPELINE")

if __name__ == "__main__":
    logger.info("Pipeline started")

    config = load_config()
    intents_path = config["paths"]["intents_path"]
    model_name = config["llm"]["model_name"]

    loader = IntentJSONLoader(intents_path)
    intents_data = loader.load()

    df = flatten_intents_json(intents_data)
    logger.info(f"Flattened dataset with {len(df)} records")

    df = df.sample(n=25, random_state=42).reset_index(drop=True)
    logger.info("Sampled 25 records for evaluation")

    nlu = GemmaNLU(model_name)

    predictions = []

    for i, row in df.iterrows():
        logger.info(f"[{i+1}/{len(df)}] Processing text: {row['text']}")
        result = nlu.predict(row["text"], intents_data)
        predictions.append(result["intent"])

    df["predicted_intent"] = predictions
    logger.info("Prediction completed")

    evaluator = NLUEvaluator()
    evaluator.evaluate(df)

    logger.info("Pipeline finished successfully")
