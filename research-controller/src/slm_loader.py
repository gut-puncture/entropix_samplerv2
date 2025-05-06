"""
Handles loading the frozen Small Language Model (SLM) and its tokenizer.

Responsibilities:
- Load the specified SLM (e.g., Phi-3-Mini) with quantization (4-bit).
- Load the corresponding tokenizer.
- Add specified special tokens to the tokenizer and resize model embeddings.
- Persist the extended tokenizer for consistent use across runs.
- Return the model (in eval mode, no gradients) and the tokenizer.
- Ensure only one instance of the model/tokenizer is loaded globally (optional optimization).
"""

from typing import Tuple

# Assuming transformers is installed. We'll add specific imports later.
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Placeholder types until transformers is used
ModelType = object 
TokenizerType = object

# Global variables to hold the single instance (optional optimization)
_slm_model = None
_slm_tokenizer = None

def load_slm(force_reload: bool = False) -> Tuple[ModelType, TokenizerType]:
    """Loads the quantized SLM and its tokenizer with special tokens added.

    Implements a singleton pattern (optional) to avoid reloading the model
    unless `force_reload` is True.

    Args:
        force_reload: If True, forces reloading the model and tokenizer
                      even if they are already loaded.

    Returns:
        A tuple containing:
            - model: The loaded language model (e.g., a transformers PreTrainedModel).
                     In eval mode with gradients disabled.
            - tokenizer: The loaded tokenizer (e.g., a transformers PreTrainedTokenizer).
                         Extended with special tokens.

    Raises:
        ImportError: If required libraries (transformers, torch, bitsandbytes) 
                     are not installed.
        FileNotFoundError: If the specified tokenizer path for saving/loading doesn't work.
        ValueError: If the configuration specifies an invalid model or quantization setup.
        # Add other potential exceptions like huggingface_hub errors.
    """
    global _slm_model, _slm_tokenizer

    # TODO: Check if model/tokenizer already loaded and return if not force_reload
    
    # TODO: Load configuration using `from src.config import cfg`
    # model_name = cfg['slm']['model_name']
    # load_bits = cfg['slm']['load_bits']
    # tokenizer_path = cfg['slm']['tokenizer_path']
    # special_tokens = cfg['slm']['special_tokens']

    # TODO: Implement BitsAndBytesConfig for 4-bit loading
    
    # TODO: Load the model using AutoModelForCausalLM.from_pretrained
    #           - Pass quantization_config
    #           - Use device_map="cpu" (as per requirement)
    
    # TODO: Load the tokenizer using AutoTokenizer.from_pretrained

    # TODO: Add special tokens
    # num_added = tokenizer.add_tokens(special_tokens)
    # if num_added > 0:
    #     model.resize_token_embeddings(len(tokenizer))
    #     # TODO: Save the extended tokenizer to tokenizer_path

    # TODO: Set model to eval mode and disable gradients
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False

    # TODO: Store the loaded model and tokenizer in global vars (optional)
    # _slm_model = model
    # _slm_tokenizer = tokenizer

    # TODO: Return the loaded model and tokenizer
    raise NotImplementedError("SLM loading is not yet implemented.")

# Example usage (optional)
if __name__ == '__main__':
    print("Attempting to load SLM and tokenizer...")
    try:
        # Note: This will fail until the TODOs are implemented
        model, tokenizer = load_slm()
        print("SLM and Tokenizer loaded successfully (placeholder).")
        # print(f"Model type: {type(model)}")
        # print(f"Tokenizer type: {type(tokenizer)}")
        # print(f"Tokenizer vocab size: {len(tokenizer)}")
    except NotImplementedError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 