from typing import Text
from transformers import AutoTokenizer, AutoModel # type: ignore
import torch # type: ignore

class Encoder:
    def __init__(
        self,
        device: torch.device,
        model_name_or_path: Text = "answerdotai/ModernBERT-base",
    ):
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
    
    def encode(
        self,
        input_text: Text
    ):
        if isinstance(input_text, str):  # Wrap single string in a list
            input_text = [input_text]
        inputs = self.tokenizer.batch_encode_plus(
            input_text,
            max_length=4096,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def load_tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer
    
    def load_model(self,):
        model = AutoModel.from_pretrained(self.model_name_or_path)
        return model.to(self.device)