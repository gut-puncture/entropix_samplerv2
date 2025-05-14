from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import os
try:
    import psutil
except ImportError:
    psutil = None

model_path = "assets/phi-3-mini-4k-instruct-model"
prompt = "Hello, who are you and what can you do?"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Configure 4-bit quantization via bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

if psutil:
    proc = psutil.Process(os.getpid())
    print(f"[Memory] After quantized load RSS={proc.memory_info().rss/(1024**2):.1f}MB")

# Try compiling the model with torch.compile (PyTorch 2.0+) for faster execution
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model, backend="inductor")
        print("Model compiled with torch.compile (inductor).")
    except Exception as e:
        print(f"torch.compile failed: {e}")

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=30,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False
    )

if psutil:
    proc = psutil.Process(os.getpid())
    print(f"[Memory] After generation RSS={proc.memory_info().rss/(1024**2):.1f}MB")

print("Prompt:", prompt)
print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True)) 