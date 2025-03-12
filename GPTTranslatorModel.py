import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class GPTTranslatorConfig(PretrainedConfig):
    model_type = "gpt_translator"
    def __init__(self, d_model=768, num_heads=12, num_layers=12, dim_feedforward=3072, max_seq_len=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len

# Define GPT-like Transformer Model
class GPTTranslatorModel(PreTrainedModel):
    def __init__(self, tokenizer_config, config):
        vocab_size = tokenizer_config['vocab_size']
        super().__init__(config)
        self.eos_id = tokenizer_config['eos_id']
        # self.bos_id = tokenizer_config['bos_id']
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation='gelu',
            dropout=0.2,
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.fc_out = nn.Linear(config.d_model, vocab_size)

        self.max_seq_len = config.max_seq_len
        self.register_buffer("causal_mask", torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), -1e4),
            diagonal=1
        ).bool())
        self.output_norm = nn.LayerNorm(config.d_model)



    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max allowed {self.max_seq_len}")

        # Token and positional embeddings
        token_embeds = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)
        pos_embeds = self.positional_embedding(positions)
        embeddings = token_embeds + pos_embeds
        embeddings = self.input_norm(embeddings)
        # Slice the pre-registered causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len].to(input_ids.device)
        
        # Transformer encoder with causal masking
        output = self.transformer(embeddings, embeddings, tgt_mask=causal_mask)
        output = self.output_norm(output + embeddings)

        return self.fc_out(output)
    
    def generate(self, input_ids, max_length=512, temperature=1.0, top_k=50):
        self.eval()
        # device = input_ids.device
        # batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        eos_token_id = self.eos_id

        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            logits = self(generated)  
            
            # Process logits
            next_logits = logits[:, -1, :] / temperature
            top_k_values = torch.topk(next_logits, top_k, dim=-1)[0]
            next_logits[next_logits < top_k_values[:, -1:]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
                        
            # Stop condition
            if (next_token == eos_token_id).all():
                break
                
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated
    
    def _forward_decoder(self, tgt, memory, mask):
        return self.transformer(tgt, memory, tgt_mask=mask)
