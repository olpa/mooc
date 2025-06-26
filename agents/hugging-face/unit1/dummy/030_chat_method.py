import os
from huggingface_hub import InferenceClient

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")


output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
