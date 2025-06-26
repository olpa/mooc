import os
from huggingface_hub import InferenceClient

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")

prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print(output)
