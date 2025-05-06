"""
Handles the loading of the frozen Small Language Model (SLM) and its tokenizer.
"""

from typing import Tuple

# Attempt to import PyTorch and Transformers, but allow for placeholder if not yet installed
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # Specify the return type more accurately if torch.nn.Module is too general later
    ModelType = torch.nn.Module 
    TokenizerType = AutoTokenizer
except ImportError:
    print("Warning: PyTorch or Transformers not fully installed. SLM loader will use placeholders.")
    # Define placeholder types if imports fail, useful for initial skeleton setup
    class PlaceholderModel:
        pass
    class PlaceholderTokenizer:
        def __init__(self, special_tokens_map=None):
            self.special_tokens_map = special_tokens_map or {}
        def encode(self, text):
            return [0, 1, 2] # Dummy encoding
        def decode(self, tokens):
            return "dummy text" # Dummy decoding
        def add_special_tokens(self, special_tokens_dict):
            pass
        @property
        def pad_token_id(self):
            return 0

    ModelType = PlaceholderModel
    TokenizerType = PlaceholderTokenizer 

from config import cfg

def load_slm() -> Tuple[ModelType, TokenizerType]:
    """
    Loads the specified Small Language Model (SLM) and its tokenizer.

    The function handles:
    - Fetching the model and tokenizer from Hugging Face based on `cfg.slm.model_name`.
    - Applying quantization (e.g., 4-bit using `bitsandbytes`) as per `cfg.slm.quantization`.
    - Adding special tokens (`<assistant_thought>`, `</assistant_thought>`) to the tokenizer
      and resizing the model's token embeddings accordingly.
    - Setting the model to evaluation mode (`model.eval()`) and disabling gradients (`torch.no_grad()`).

    Returns:
        Tuple[torch.nn.Module, PreTrainedTokenizer]: A tuple containing:
            - model: The loaded and configured language model.
            - tokenizer: The configured tokenizer associated with the model.
    
    Raises:
        ImportError: If essential libraries like PyTorch or Transformers are not installed
                     (unless placeholders are active due to initial setup).
        Exception: For other model loading or configuration errors.
    """
    # TODO: Step 3: Fill implementation (quantised load, token resizing, no-grad)
    # This is just a skeleton for now.
    
    print(f"Attempting to load SLM: {cfg.slm.model_name}")
    print(f"Quantization: {cfg.slm.quantization}")

    # --- Placeholder Implementation (to be replaced) ---
    if 'PlaceholderModel' in globals(): # Check if we are in placeholder mode
        print("SLM Loader: Running in placeholder mode.")
        # Create placeholder instances for tokenizer and model
        tokenizer = TokenizerType(
            special_tokens_map= {
                "pad_token": "[PAD]", 
                cfg.slm.special_tokens.assistant_thought_begin: cfg.slm.special_tokens.assistant_thought_begin,
                cfg.slm.special_tokens.assistant_thought_end: cfg.slm.special_tokens.assistant_thought_end
            }
        )
        model = ModelType()
        return model, tokenizer
    # --- End Placeholder Implementation ---

    # Actual implementation will go here in Step 3 of Phase 2
    # For now, to allow the test to run, we'll raise a NotImplementedError if not in placeholder mode.
    raise NotImplementedError("SLM loading (model and tokenizer) is not yet implemented.")

if __name__ == "__main__":
    # Example usage (will use placeholders or raise NotImplementedError initially)
    try:
        model, tokenizer = load_slm()
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        if not isinstance(model, PlaceholderModel):
            # Basic test with special tokens if actual tokenizer is loaded
            special_tokens_dict = {
                'additional_special_tokens': [
                    cfg.slm.special_tokens.assistant_thought_begin,
                    cfg.slm.special_tokens.assistant_thought_end
                ]
            }
            # tokenizer.add_special_tokens(special_tokens_dict) # This would be part of the actual load_slm
            
            thought_begin = cfg.slm.special_tokens.assistant_thought_begin
            thought_end = cfg.slm.special_tokens.assistant_thought_end
            
            encoded_begin = tokenizer.encode(thought_begin, add_special_tokens=False)
            decoded_begin = tokenizer.decode(encoded_begin, skip_special_tokens=False)
            print(f"Encoded '{thought_begin}': {encoded_begin} -> Decoded: '{decoded_begin}'")

            encoded_end = tokenizer.encode(thought_end, add_special_tokens=False)
            decoded_end = tokenizer.decode(encoded_end, skip_special_tokens=False)
            print(f"Encoded '{thought_end}': {encoded_end} -> Decoded: '{decoded_end}'")

    except Exception as e:
        print(f"Error during example usage: {e}") 