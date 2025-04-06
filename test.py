from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# models = client.models.list()
# for model in models.data:
    # print(f"ID: {model.id}, Name: {model.id}")

# List all models
models = client.models.list()

# Filter high-context, free models (adjust keywords as needed)
high_context_models = []
for model in models.data:
    model_id = model.id.lower()
    if any(keyword in model_id for keyword in ["llama", "mistral", "deepseek", "qwen", "zephyr", "openchat","gemini"]):
        high_context_models.append(model_id)

# Print selected models
for model in high_context_models:
    print('"', model,'"',",",sep="")



# OPENROUTER_MODELS = [
# "openai/gpt-3.5-turbo",
# "meta-llama/llama-4-maverick:free",
# "meta-llama/llama-4-scout:free",
# "mistral/ministral-8b",
# "deepseek/deepseek-v3-base:free",
# "scb10x/llama3.1-typhoon2-70b-instruct",
# "google/gemini-2.5-pro-exp-03-25:free",
# "qwen/qwen2.5-vl-32b-instruct:free",
# "deepseek/deepseek-chat-v3-0324:free",
# "mistralai/mistral-small-3.1-24b-instruct:free",
# "deepseek/deepseek-r1-zero:free",
# "qwen/qwq-32b:free",
# "nousresearch/deephermes-3-llama-3-8b-preview:free",
# "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
# "deepseek/deepseek-r1-distill-llama-8b",
# "qwen/qwen-vl-max",
# "qwen/qwen2.5-vl-72b-instruct:free",
# "mistralai/mistral-small-24b-instruct-2501:free",
# "deepseek/deepseek-r1-distill-qwen-32b:free",
# "deepseek/deepseek-r1-distill-llama-70b:free",
# "google/gemini-2.0-flash-thinking-exp:free",
# "deepseek/deepseek-r1:free",
# "deepseek/deepseek-chat:free",
# "google/gemini-2.0-flash-exp:free",
# "meta-llama/llama-3.3-70b-instruct:free",
# "qwen/qwq-32b-preview:free",
# "nvidia/llama-3.1-nemotron-70b-instruct:free",
# "meta-llama/llama-3.2-3b-instruct:free",
# "meta-llama/llama-3.2-1b-instruct:free",
# "qwen/qwen-2.5-72b-instruct:free",
# "meta-llama/llama-3.1-405b",
# "perplexity/llama-3.1-sonar-large-128k-online",
# "meta-llama/llama-3.1-8b-instruct:free",
# "meta-llama/llama-3.1-70b-instruct",
# "mistralai/codestral-mamba",
# "mistralai/mistral-nemo:free",
# "mistralai/mistral-7b-instruct:free",
# "openchat/openchat-7b:free",
# "huggingfaceh4/zephyr-7b-beta:free",
# "meta-llama/llama-2-13b-chat",
# "meta-llama/llama-2-70b-chat"]
