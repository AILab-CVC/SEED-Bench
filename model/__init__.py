MODEL_REGISTRY={
    "instructblip": {
        "model_name": "Salesforce/instructblip-flan-t5-xl",
        "weight_name": "Salesforce/instructblip-flan-t5-xl",
        "model_type": "mllm"
        }
}

def get_model_info(model_name):
    return MODEL_REGISTRY[model_name]
