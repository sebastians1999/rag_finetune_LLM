import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

class ModelEvaluator:
    def __init__(self, base_model_name, finetuned_model_name):
        print("Loading models...")

        # Tokenizer (identical for both models)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)

        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Base model
        print("Loading base model in 4-bit...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.print_model_size(self.base_model, "Base Model")

        # Finetuned model
        print("Loading fine-tuned model in 4-bit...")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.print_model_size(self.finetuned_model, "Fine-Tuned Model")

        print("Models loaded successfully!\n")

    def print_model_size(self, model, name: str):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate memory footprint
        param_bytes = 0
        for p in model.parameters():
            if hasattr(p, "quant_state"):  # bitsandbytes quantized
                param_bytes += p.numel() * 0.5  
            else:
                param_bytes += p.numel() * p.element_size()

        size_mb = param_bytes / (1024**2)

        print(f"\n[{name}]")
        print(f"  • Total parameters:     {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print(f"  • Approx size:          {size_mb:.2f} MB\n")

    def calculate_perplexity(self, model, texts):
        model.eval()
        total_loss = 0
        total_tokens = 0

        print("Calculating perplexity...")
        for text in tqdm(texts):
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc.input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)

        ppl = torch.exp(torch.tensor(total_loss / total_tokens))
        return ppl.item()

def main():
    BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
    FINETUNED_MODEL = "schmuelling/Llama-3.2-1B-Instruct-finetome"

    print("Loading dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:100]")

    test_texts = [
        item["conversations"][0]["value"]
        for item in dataset
        if len(item["conversations"]) > 0
    ][:50]

    evaluator = ModelEvaluator(BASE_MODEL, FINETUNED_MODEL)

    print("\n=== PERPLEXITY EVALUATION ===")
    base_ppl = evaluator.calculate_perplexity(evaluator.base_model, test_texts)
    ft_ppl = evaluator.calculate_perplexity(evaluator.finetuned_model, test_texts)

    improvement = ((base_ppl - ft_ppl) / base_ppl) * 100

    print(f"\nBase Model Perplexity:       {base_ppl:.2f}")
    print(f"Fine-Tuned Model Perplexity: {ft_ppl:.2f}")
    print(f"Improvement:                 {improvement:.2f}%")

if __name__ == "__main__":
    main()
