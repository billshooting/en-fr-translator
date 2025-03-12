import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import TranslationDataset
import GPTTranslatorModel
from datetime import datetime
import os
from TrainingVisualizer import TrainingVisualizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  

def main():
    # hardware
    device = "mps" if torch.accelerator.is_available() else "cpu"
    torch.set_float32_matmul_precision('high') 

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})
    tokenizer.pad_token = '[PAD]'
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = TranslationDataset.TranslationDataset(
        './data/Wikimedia-20230407/wikimedia.en-fr.en', 
        './data/Wikimedia-20230407/wikimedia.en-fr.fr',
        tokenizer,
        512)
    train_size = int(0.04 * len(dataset))
    train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2)

    vocab_size = len(tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    print(f"vocab size is {vocab_size}, eos id is {eos_id}")
    tokenizer_config = { 'vocab_size': vocab_size, 'eos_id': eos_id }
    model = GPTTranslatorModel.GPTTranslatorModel(tokenizer_config, GPTTranslatorModel.GPTTranslatorConfig()).to(device)
    # model = torch.compile(model)  # Graph compilation
    model.config.use_cache = False  # Disable cache for training
    # model = model.to(torch.float32)  # MPS-optimized precision
    optimizer = optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.001)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=0.01).to(device)
    train(model, train_data_loader, optimizer, criterion, tokenizer, 3, 4)

    # model.eval()

    model_save_path = "gpt_like_translator.pth"
    model_save_all = "gpt_like_translator_model-0.1"
    tokenizer_save_path = "gpt_neo_tokenizer"
    model.save_pretrained(model_save_all)
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")

# Training loop
def train(model, dataloader, optimizer, criterion, tokenizer, epochs=3, grad_accum_steps=4):
    device = "mps" if torch.accelerator.is_available() else "cpu"
    model.train()

    visualizer = TrainingVisualizer()
    global_step = 0
    total_steps = len(dataloader) * epochs

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=4e-4,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    for epoch in range(epochs):
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
        # 6a. Mixed Precision Training
            with torch.autocast(device_type='mps', dtype=torch.bfloat16):
                input_ids, labels = [b.to(device, non_blocking=True) for b in batch]
                outputs = model(input_ids)
                # Ensure loss is only calculated on target tokens
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                # Calculate loss only on target tokens (labels != -100)
                # loss = criterion(
                #     outputs.view(-1, outputs.size(-1)),
                #     labels.view(-1))
                loss = loss / grad_accum_steps

                if step % 50 == 0:
                    log_input = input_ids[0]
                    log_label = labels[0]
                    log_outout = outputs[0]
                    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
                    sep_pos = (log_input == sep_token_id).nonzero(as_tuple=True)[0]
                    if sep_pos.numel() > 0:
                        input_str = tokenizer.decode(log_input, skip_special_tokens=False)
                        log_label_token = log_label[log_label != -100]
                        label_str = tokenizer.decode(log_label_token, skip_special_tokens=False)

                        pred_tokens = log_outout.argmax(dim=-1)
                        pred_after_sep = pred_tokens[sep_pos:]
                        predict_str = tokenizer.decode(pred_after_sep, skip_special_tokens=False)
                        print(f"\n--- Step {step} ---")
                        print(f"Input Prompt:\n{input_str}")
                        print(f"\nActual Translation:\n{label_str}")
                        print(f"\nPredicted Translation:\n{predict_str}")
                        print("-----------------------\n")

            # 6b. Gradient Accumulation
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # 6c. Progress Tracking
            if step % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                visualizer.update(global_step, loss.item() * grad_accum_steps, current_lr)
                mem_usage = torch.mps.current_allocated_memory() / 1e9
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss: {loss.item() * grad_accum_steps:.3f} | "
                    f"LR: {current_lr:.1e} | "
                    f"Mem: {mem_usage:.1f}GB"
                )
            # monitor the gradient
            if step % 50 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient Norm: {total_norm:.4f}")

            global_step += 1
        print(f"Epoch {epoch+1} Completed | Avg Loss: {loss.item() * grad_accum_steps:.3f}")
        visualizer.save_plots()


if __name__ == '__main__':
    # Required for multiprocessing on macOS
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()