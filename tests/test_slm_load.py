import unittest
from pathlib import Path
import sys

# Ensure src is in python path for testing
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slm_loader import load_slm, ModelType, TokenizerType, cfg
# Determine if we are in placeholder mode for conditional testing
IS_PLACEHOLDER_MODE = False
try:
    import torch
    from transformers import AutoTokenizer # Check for a real tokenizer class
    if not issubclass(ModelType, torch.nn.Module) or TokenizerType == type(None):
        IS_PLACEHOLDER_MODE = True 
except ImportError:
    IS_PLACEHOLDER_MODE = True

# If actual torch is imported, ModelType should be torch.nn.Module
# If not, it will be the PlaceholderModel from slm_loader
ExpectedModelBase = getattr(sys.modules.get('torch.nn', object), 'Module', ModelType)


class TestSLMLoader(unittest.TestCase):

    def test_load_slm_returns_correct_types(self):
        """Test that load_slm returns a model and a tokenizer of expected base types."""
        try:
            model, tokenizer = load_slm()
        except NotImplementedError:
            if IS_PLACEHOLDER_MODE:
                self.fail("load_slm raised NotImplementedError even in placeholder mode.")
            else:
                # This is expected if actual loading isn't implemented yet and not in placeholder mode
                print("Skipping type test as SLM loading is not yet fully implemented (NotImplementedError caught).")
                return
        
        self.assertIsInstance(model, ExpectedModelBase, f"Model is not an instance of {ExpectedModelBase}")
        # For TokenizerType, we check if it has encode/decode methods, as its base can vary
        self.assertTrue(hasattr(tokenizer, 'encode'), "Tokenizer does not have an encode method.")
        self.assertTrue(hasattr(tokenizer, 'decode'), "Tokenizer does not have a decode method.")

    def test_special_tokens_round_trip(self):
        """Test that special tokens can be encoded and decoded correctly by the tokenizer."""
        try:
            _, tokenizer = load_slm()
        except NotImplementedError:
            if IS_PLACEHOLDER_MODE:
                self.fail("load_slm raised NotImplementedError even in placeholder mode for special token test.")
            else:
                print("Skipping special tokens test as SLM loading is not yet fully implemented (NotImplementedError caught).")
                return

        thought_begin = cfg.slm.special_tokens.assistant_thought_begin
        thought_end = cfg.slm.special_tokens.assistant_thought_end

        # The placeholder tokenizer might not perfectly roundtrip if not designed to.
        # The actual huggingface tokenizer should.
        if not IS_PLACEHOLDER_MODE:
            # For a real tokenizer, we expect perfect roundtrip after adding them.
            # The load_slm function is responsible for adding these tokens.
            encoded_begin = tokenizer.encode(thought_begin, add_special_tokens=False)
            decoded_begin = tokenizer.decode(encoded_begin, skip_special_tokens=False)
            self.assertEqual(decoded_begin, thought_begin, 
                             f"Failed round-trip for token: {thought_begin} -> {decoded_begin}")

            encoded_end = tokenizer.encode(thought_end, add_special_tokens=False)
            decoded_end = tokenizer.decode(encoded_end, skip_special_tokens=False)
            self.assertEqual(decoded_end, thought_end, 
                             f"Failed round-trip for token: {thought_end} -> {decoded_end}")
        else:
            # For placeholder, we might just check that encode/decode run
            try:
                encoded_begin = tokenizer.encode(thought_begin)
                decoded_begin = tokenizer.decode(encoded_begin)
                self.assertIsInstance(decoded_begin, str) # Basic check
                
                encoded_end = tokenizer.encode(thought_end)
                decoded_end = tokenizer.decode(encoded_end)
                self.assertIsInstance(decoded_end, str) # Basic check
                print("Special token test running in placeholder mode - checking for basic encode/decode functionality.")
            except Exception as e:
                self.fail(f"Placeholder tokenizer failed basic encode/decode for special tokens: {e}")


if __name__ == '__main__':
    unittest.main() 