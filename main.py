import torch
import tiktoken
from src_codes.gpt import GPTModel

# Model configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


# Load quantized model
def load_model(model_path: str):
    print("Loading quantized model...")

    # Quantized models run best on CPU
    device = torch.device("cpu")

    # Build model structure
    model = GPTModel(GPT_CONFIG_355M)

    # Load already-quantized weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print("✓ Model loaded successfully!")
    return model, device


# Token helpers
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# Text generation
def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# Main program
def main():
    model_path = "model/gpt2-medium355M-sft-quantized.pth"
    model, device = load_model(model_path)

    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = input(
        "Enter start context for text generation: "
    )

    input_ids = text_to_token_ids(
        start_context, tokenizer
    ).to(device)

    print("Generating text...")

    token_ids = generate_text(
        model=model,
        idx=input_ids,
        max_new_tokens=20,
        context_size=GPT_CONFIG_355M["context_length"],
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\nOutput text:\n")
    print(output_text)


if __name__ == "__main__":
    main()
