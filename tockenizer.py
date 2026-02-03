from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, how are you?"

# Basic tokenization
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("original text:", text)
print("tokens:", tokens)
print("token ids:", token_ids)

# Encode method (does tokenization + conversion to ids)
encoded = tokenizer(text)
print("Encoded text:", encoded)

# Decoding
decoded = tokenizer.decode(encoded["input_ids"])
print("Decoded text:", decoded)