from typing import Text
from transformers import AutoTokenizer, ModernBertModel # type: ignore
import torch # type: ignore

class Encoder:
    def __init__(
        self,
        model_name_or_path: Text = "answerdotai/ModernBERT-base"
    ):
        self.model_name_or_path = model_name_or_path
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
    
    def encode(
        self,
        input_text: Text
    ):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
    
    def load_tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer
    
    def load_model(self,):
        model = ModernBertModel.from_pretrained(self.model_name_or_path)
        return model