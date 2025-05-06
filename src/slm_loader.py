"""
Handles the loading of the frozen Small Language Model (SLM) and its tokenizer.
"""

from typing import Tuple
from pathlib import Path

# Attempt to import PyTorch and Transformers, but fallback to placeholder on any failure
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # Specify the return type more accurately if torch.nn.Module is too general later
    ModelType = torch.nn.Module
    TokenizerType = AutoTokenizer
except Exception as e:
    print(f"Warning: Could not import PyTorch or Transformers ({e}). SLM loader will use placeholders.")
    # Define placeholder types if imports fail, useful for initial skeleton setup
    try:
        BaseModel = torch.nn.Module
    except Exception:
        BaseModel = object
    class PlaceholderModel(BaseModel):
        """A placeholder model for environments without torch/transformers."""
        pass
    class PlaceholderTokenizer:
        """A placeholder tokenizer with basic encode/decode methods."""
        def __init__(self, special_tokens_map=None):
            self.special_tokens_map = special_tokens_map or {}
        def encode(self, text, add_special_tokens=False):
            return [0, 1, 2]  # Dummy encoding
        def decode(self, tokens, skip_special_tokens=False):
            return "dummy text"  # Dummy decoding
        def add_special_tokens(self, special_tokens_dict):
            pass
        @property
        def pad_token_id(self):
            return 0

    ModelType = PlaceholderModel
    TokenizerType = PlaceholderTokenizer 

from config import cfg # Use the global cfg instance

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

    # Actual implementation: load model via MPS float16 fallback or dynamic quantization on CPU
    if torch.backends.mps.is_available():
        print("MPS device available. Loading model in float16 on MPS.")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.slm.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to("mps")
    else:
        print("No MPS device available. Loading model with dynamic quantization on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.slm.model_name,
            low_cpu_mem_usage=True
        )
        print("Applying dynamic quantization to linear modules.")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    # 3. Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.slm.model_name, use_fast=True)
    special_tokens = [
        cfg.slm.special_tokens.assistant_thought_begin,
        cfg.slm.special_tokens.assistant_thought_end
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 4. Resize token embeddings and save tokenizer
    model.resize_token_embeddings(len(tokenizer))
    save_dir = Path("assets/tokenizer_extended")
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)

    # 5. Freeze model and set to eval mode
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = load_slm()
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")

        print("\n--- Attempting Generation with Loaded Model ---")
        try:
            import torch
            prompt = "Hello, can you tell me a short story?"
            print(f"Prompt: {prompt}")

            # Tokenize and prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to the same device as model
            device = next(model.parameters()).device
            inputs = inputs.to(device)

            print("Generating response...")
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50
                )
            response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            print(f"Generated Response: {response}")
        except ImportError:
            print("Torch is not available, cannot perform generation test.")
        except Exception as e:
            print(f"Error during generation test: {e}")
        print("--- End Generation Test ---")
    except Exception as e:
        print(f"Error during example usage: {e}") 