import torch


def generate(model, prompt_ids, max_tokens, eos_token_id):
    input_ids = prompt_ids.clone()
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
        if next_token.item() == eos_token_id:
            break
    return input_ids
