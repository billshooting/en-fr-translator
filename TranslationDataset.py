from torch.utils.data import Dataset
import re

class TranslationDataset(Dataset):
    def __init__(self, en_file, fr_file, tokenizer, max_length):
        super(TranslationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(en_file, 'r', encoding='utf-8') as ef:
            self.english_sentences = ef.readlines()
        with open(fr_file, 'r', encoding='utf-8') as ff:
            self.french_sentences = ff.readlines()

        self.english_sentences = [re.sub(r'\[.*?\]', '', s).strip() for s in self.english_sentences]
        self.french_sentences = [re.sub(r'\[.*?\]', '', s).strip() for s in self.french_sentences]

    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, index):
        src_text = self.english_sentences[index]
        tgt_text = self.french_sentences[index] + ' ' + self.tokenizer.eos_token
        
        # Combine source + target with a separator
        full_text = f"Translate English to French: {src_text} [SEP] {tgt_text}"
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Safely find [SEP] position
        sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        sep_mask = (encoding.input_ids == sep_token_id)
        
        if sep_mask.any():
            sep_pos = sep_mask.nonzero()[0, 1].item() 
        else:
            sep_pos = self.max_length - 1
        
        # Mask loss for source + [SEP]
        labels = encoding.input_ids.clone()
        labels[:, :sep_pos+1] = -100
        
        return encoding.input_ids.squeeze(0), labels.squeeze(0)