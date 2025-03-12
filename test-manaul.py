import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer
from datetime import datetime
from GPTTranslatorModel import GPTTranslatorModel, GPTTranslatorConfig
from safetensors.torch import load_file


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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

model.eval()

def translate_text(input_text, target_text = ''):
    # 1. 构造输入
    prompt = f"Translate English to French: {input_text} [SEP] {target_text}"
    input_ids = tokenizer.encode(
        prompt, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=512
    ).to(device)
    
    # 2. 前向传播
    with torch.no_grad():
        logits = model(input_ids)
    
    # 3. 提取 [SEP] 后的 logits
    sep_pos = (input_ids == sep_id).nonzero()[0, 1].item()
    target_logits = logits[:, sep_pos+1:, :]
    
    # 4. 转换为 token ID
    predicted_ids = torch.argmax(target_logits, dim=-1)
    all_ids = torch.argmax(logits, dim=-1)
    # 5. 解码并清理
    translated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
    all_text = tokenizer.decode(all_ids[0], skip_special_tokens=False)
    
    # 6. 截断到第一个 EOS（如果有）
    if tokenizer.eos_token in translated_text:
        translated_text = translated_text.split(tokenizer.eos_token, 1)[0]
    
    return translated_text, all_text

a, b  = translate_text("Around the same time, Julius Watkins joined a six-member jazz band playing horn.", "À la même époque,")
print(f"translated: {a}")
print ("---------------")
print(f"full: {b}")
# À la même époque, Julius Watkins rejoint un sextet de jazz au cor.