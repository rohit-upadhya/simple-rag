from typing import Text, Optional
import torch # type: ignore
from datasets import load_dataset # type: ignore

from src.encoder import Encoder
class Rag:
    def __init__(
        self,
        input_file: Optional[Text] = None,
        device: Text = "cpu",
    ):
        self.input_file = input_file
        self.device = torch.device(device)
        self.encoder = Encoder(device)
        self.load_input()

    def load_input(self,):
        rag_dataset = load_dataset("neural-bridge/rag-dataset-12000")
        self.dataset = rag_dataset["train"]
    
    def encode_dataset(self, datapoints, batch_size = 256):
        
        pass
    
    def api_caller(self,):
        
        pass
    
    def chat_bot(self,):
        
        pass

if __name__=="__main__":
    rag = Rag()