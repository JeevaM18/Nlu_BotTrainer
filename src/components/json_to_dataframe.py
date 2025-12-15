import pandas as pd

def flatten_intents_json(intents_data):
    rows = []

    for intent in intents_data["intents"]:
        for example in intent["examples"]:
            rows.append({
                "text": example,
                "true_intent": intent["name"]
            })

    df = pd.DataFrame(rows)
    return df
