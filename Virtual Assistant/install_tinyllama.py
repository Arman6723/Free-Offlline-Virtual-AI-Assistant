from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def install_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Downloading TinyLlama-1.1B-Chat model (~2GB) from {model_name}...")
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer downloaded successfully.")
        # Download model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading TinyLlama: {str(e)}")
        return
    print("TinyLlama-1.1B-Chat is ready. Run 'python main.py' to start the AI Assistant.")

if __name__ == "__main__":
    install_tinyllama()