from torch.utils.data import Dataset
import re
from typing import Dict, Tuple
import torch

class TranslationDataProcessor:
    """Data processing class for handle data cleaning and masking"""
    
    def __init__(self, 
                 tokenizer,
                 max_length: int = 512,
                 task_prefix: str = "Translate English to French:",
                 sep_token: str = "[SEP]",
                 eos_token: str = "<EOS>"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = task_prefix
        self.sep_token = sep_token
        self.eos_token = eos_token
        
        self._validate_special_tokens()
        
    def _validate_special_tokens(self):
        required_tokens = [self.sep_token, self.eos_token]
        for token in required_tokens:
            if token not in self.tokenizer.additional_special_tokens:
                raise ValueError(f"'{token}' must be added to tokenizer's special tokens")

    def clean_text(self, text: str) -> str:
        # remove citation content
        text = re.sub(r'\[.*?\]', '', text) 
        return text.strip()
    
    def format_input_text(self, src_text: str) -> str:
        return f"{self.task_prefix} {src_text} {self.sep_token}"
    
    def format_target_text(self, tgt_text: str) -> str:
        return f"{tgt_text} {self.eos_token}"
    
    def encode_sample(self, src_text: str, tgt_text: str) -> Dict[str, torch.Tensor]:
        # 1. clean the text
        clean_src = self.clean_text(src_text)
        clean_tgt = self.clean_text(tgt_text)
        
        # 2. construct the full sequence
        full_text = f"{self.format_input_text(clean_src)} {self.format_target_text(clean_tgt)}"
        
        # 3. tokenized and encoding
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 4. masking
        labels = self._create_labels_mask(encoding.input_ids)
        
        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def _create_labels_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """creating the loss masking: only calculate loss on generated content"""
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        sep_positions = (input_ids == sep_token_id).nonzero()
        
        if sep_positions.nelement() == 0:
            sep_pos = self.max_length - 1 
        else:
            sep_pos = sep_positions[0, 1].item()
        
        labels = input_ids.clone()
        labels[:, :sep_pos+1] = -100  # mask the input
        
        # mask the content after <EOS>
        eos_pos = (input_ids == self.tokenizer.eos_token_id).nonzero()
        if eos_pos.nelement() > 0:
            labels[:, eos_pos[0,1]+1:] = -100
            
        return labels

class TranslationDataset(Dataset):
    
    def __init__(self, 
                 en_file: str, 
                 fr_file: str, 
                 data_processor: TranslationDataProcessor):
        self.processor = data_processor
        
        with open(en_file, 'r', encoding='utf-8') as f:
            self.english_samples = [self._clean_line(line) for line in f]
            
        with open(fr_file, 'r', encoding='utf-8') as f:
            self.french_samples = [self._clean_line(line) for line in f]
            
        if len(self.english_samples) != len(self.french_samples):
            raise ValueError("English and French files have different number of sentences")
            
    def _clean_line(self, line: str) -> str:
        return line.strip()
    
    def __len__(self) -> int:
        return len(self.english_samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.english_samples[index]
        tgt = self.french_samples[index]
        return self.processor.encode_sample(src, tgt)