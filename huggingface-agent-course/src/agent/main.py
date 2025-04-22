# Contents of /huggingface-agent-course/huggingface-agent-course/src/agent/main.py

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

# Get the token from the .env file
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is not set in the .env file")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=hf_token)
# if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.2-3B-Instruct
# client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud", token=hf_token)


prompt= "The capital of France is"

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": prompt},
    ],
    stream=False,
    max_tokens=1024,
)

print(output.choices[0].message.content)