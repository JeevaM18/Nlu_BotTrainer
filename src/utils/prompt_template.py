def build_nlu_prompt(user_text, intents_data):

    intent_names = [intent["name"] for intent in intents_data["intents"]]

    entity_schema = intents_data["entities"]

    prompt = f"""
You are a strict NLU engine.

ALLOWED INTENTS:
{intent_names}

ENTITY SCHEMA:
{entity_schema}

Rules:
- Choose ONE intent from the allowed list only
- Extract entities ONLY from the ENTITY SCHEMA
- If an entity is not present, do not include it
- Entity values must be strings
- Do NOT hallucinate entities
- Return ONLY valid JSON (no explanation)

OUTPUT FORMAT:
{{
  "intent": "<intent_name_or_unknown>",
  "confidence": <float between 0 and 1>,
  "entities": {{
    "<entity_name>": "<entity_value>"
  }}
}}

User input:
"{user_text}"
"""
    return prompt
