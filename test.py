import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer
from datetime import datetime
from GPTTranslatorModel import GPTTranslatorModel, GPTTranslatorConfig
from safetensors.torch import load_file

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

dataset = load_dataset('wmt14', 'fr-en', split='test', cache_dir='./data/wmt14')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})
tokenizer.pad_token = '[PAD]'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# load model
model_path = './gpt_like_translator_model-0.1'
vocab_size =len(tokenizer)
config = GPTTranslatorConfig.from_json_file(f'{model_path}/config.json')
state_dict = load_file(f"{model_path}/model.safetensors")

eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
tokenizer_config = { 'vocab_size': vocab_size, 'eos_id': eos_id }

model = GPTTranslatorModel(tokenizer_config, config)
model.load_state_dict(state_dict)
model.to(device)
# model.load_state_dict(torch.load("./gpt_like_translator.pth"))

model.eval()

# load benchmark
bleu = evaluate.load("sacrebleu")

# Evaluation loop
def translate_text(input_text):
    """Translate English text to French using the GPT-like model."""
    input_text = f"Translate English to French: {input_text} [SEP]"
    input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"]
    input_ids = input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512, temperature=0.7, top_k=50)
        # predicted_ids = output_logits.argmax(dim=-1)
        # predicted_ids = torch.argmax(output_logits, dim=-1)
        # print(output_logits)
    # sep_pos = (output_ids == sep_id).nonzero()
    # if sep_pos.numel() > 0:
    #     translated_ids = output_ids[:, sep_pos[0,1]+1:]
    # else:
    #     translated_ids = output_ids
    translated_ids = output_ids
    return tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# Run translation and evaluation
predictions = []
references = []

for example in dataset:
    en_text = example["translation"]["en"]
    fr_text = example["translation"]["fr"]
    
    translated_text = translate_text(en_text)
    
    predictions.append(translated_text)
    references.append([fr_text])  # BLEU requires a list of references


for i in range(5):  # Check 5 random samples
    print(f"🔹 English: {dataset[i]['translation']['en']}")
    print(f"✅ French: {dataset[i]['translation']['fr']}")
    print(f"⚠️ French': {translate_text(dataset[i]['translation']['en'])}\n")

# Compute BLEU score
score = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {score['score']:.2f}")