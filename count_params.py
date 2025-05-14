import torch
from transformers import AutoConfig, AutoModelForCausalLM

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load the config from your config.json
config = AutoConfig.from_pretrained("deepseek-moe-16b-base/config.json")

# Initialize the model with the config
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

# Count parameters
total_params, trainable_params = count_parameters(model)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Parameters in billions: {total_params/1e9:.2f}B") 