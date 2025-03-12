import torch
from datasets import load_dataset
import evaluate
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

class EvalConfig:
    model_path = "./gpt2-ft-translator/checkpoint-6780"  # changed for actual path
    test_data_path = "../data/wmt14"
    batch_size = 1                        # adjust for memory
    max_source_length = 120               
    max_target_length = 128               
    device = "mps" if torch.backends.mps.is_available() else "cpu"  
    num_beams = 6                         
    early_stopping = True   

def main():
    # 1. load config
    config = EvalConfig()

    # 2. load tokenizer, model
    tokenizer = get_tokenizer()
    model = get_model(config)
    model.eval()

    # model.save_pretrained('./model_trained')
    # tokenizer.save_pretrained('./model_trained')
    # return

    # 3. load dataset
    test_dataset = get_test_dataset(config)
    test_dataset = test_dataset
    print(f"test_dataset sample: {test_dataset[0]}")

    # 4. process data
    encoded_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        batch_size=config.batch_size
    )
    print(f"encoded_dataset shape: {encoded_dataset.shape}")
    print(f"encoded_dataset keys: {encoded_dataset[0].keys()}")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'translation'])
    dataloader = torch.utils.data.DataLoader(
        encoded_dataset,
        batch_size=config.batch_size,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'translation': [x['translation'] for x in batch]
        }
    )

    # 5. start evaluating
    print("Starting evaluation...")
    predictions, references = generate_translations(model, tokenizer, dataloader, config)

    # 6. load benchmark
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=references)
    
    print(f"\nFinal BLEU Score: {results['score']:.2f}")
    print(f"Details: {results}")

    # 打印样本对比
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Source: {test_dataset[i]['translation']['en']}")
        print(f"Reference: {test_dataset[i]['translation']['fr']}")
        print(f"Predicted: {predictions[i]}\n")
    return

def get_tokenizer():
    """ Get tokenizer for fine tuning """

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SEP]", "<EOS>"]})
    tokenizer.pad_token = tokenizer.eos_token

    print(f"The vocab size is {len(tokenizer)}")
    return tokenizer


def get_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"The training device is {device}")
    return device

def get_test_dataset(config: EvalConfig):
    dataset = load_dataset('wmt14', 'fr-en', split='test', cache_dir=config.test_data_path)
    return dataset

def get_model(config: EvalConfig):
    model = GPT2LMHeadModel.from_pretrained(config.model_path).to(config.device)
    return model

def preprocess_function(examples, tokenizer, config: EvalConfig):
    inputs = []
    for text in examples['translation']:
        # construct input prompt
        source = text['en']
        prompt = f"Translate English to French: {source} [SEP]"
        inputs.append(prompt)
    
    # encode input
    model_inputs = tokenizer(
        inputs,
        max_length=config.max_source_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    model_inputs['translation'] = examples['translation']
    return model_inputs 

def generate_translations(model, tokenizer, dataloader, config: EvalConfig):
    predictions = []
    references = []
    
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    eos_id = tokenizer.convert_tokens_to_ids('<EOS>')
    for batch in tqdm(dataloader, desc="Evaluating"):
        # copy to GPU
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        # generate translation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_target_length,
                num_beams=config.num_beams,
                early_stopping=config.early_stopping,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # print(f"\n ---- decoded_outputs: {decoded_outputs} \n")
        clean_translations = []
        for text in decoded_outputs:
            if '[SEP]' in text:
                translated = text.split('[SEP]', 1)[1].split('<EOS>', 1)[0]
                translated = ' '.join(translated.split()).strip() 
            else:
                translated = text.replace("Translate English to French:", "").strip()
            clean_translations.append(translated)
        
        predictions.extend(clean_translations)
        references.extend([[tgt['fr']] for tgt in batch['translation']]) 
    
    return predictions, references

if __name__ == "__main__":
    main();