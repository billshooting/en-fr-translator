import torch
from dataset import TranslationDataProcessor, TranslationDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, random_split

def main():
    # prepare
    tokenizer = get_tokenizer()
    model = get_model(len(tokenizer))
    dataset = get_dataset(tokenizer, ratio=0.2)
    device = get_device()
    model.to(device)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False 
    )

    # training
    training_args = TrainingArguments(
        output_dir="./gpt2-ft-translator",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=1000,
        logging_steps=50,
        save_steps=1000,
        use_mps_device=True,
        bf16=True,  # essential for acceleration
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # callbacks=[VisualizationCallback()]
    )

    try:
        print("Start Training...")
        trainer.train()
    except Exception as e:
        print(f"Training failed: {str(e)}")
    else:
        print("Training succeed!")

def get_tokenizer():
    """ Get tokenizer for fine tuning """

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SEP]", "<EOS>"]})
    tokenizer.pad_token = tokenizer.eos_token

    print(f"The vocab size is {len(tokenizer)}")
    return tokenizer

def get_model(vocab_size: int):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)

    print(f"The model parameter size is {model.num_parameters() / 1e6:.1f}M")
    return model
    
def get_dataset(tokenizer, ratio = 0.02):
    processor = TranslationDataProcessor(tokenizer)
    dataset = TranslationDataset(
        '../data/Wikimedia-20230407/wikimedia.en-fr.en', 
        '../data/Wikimedia-20230407/wikimedia.en-fr.fr',
        processor)
    train_size = int(ratio * len(dataset))
    train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    
    print(f"The training dataset size is {len(train_dataset)}")
    return train_dataset

def get_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"The training device is {device}")
    return device

if __name__ == '__main__':
    # Required for multiprocessing on macOS
    # torch.multiprocessing.set_start_method('spawn', force=True)
    main()