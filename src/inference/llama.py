from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

import torch
import os

env_file = ".env.dev"
load_dotenv(env_file)

login(os.getenv("HUGGING_FACE_KEY"))


class LlamaInference:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
        load_quantized: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.load_quantized = load_quantized
        self.device = device

    def _quantization_config(
        self,
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        return bnb_config

    def _load_model(
        self,
    ):
        model_config = {
            "pretrained_model_name_or_path": self.model_name_or_path,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        try:
            if self.load_quantized:
                quantization_config = self._quantization_config()
                model_config["quantization_config"] = quantization_config
            model = AutoModelForCausalLM.from_pretrained(**model_config)
        except:
            raise ValueError(
                "Issue loading local model. Use OpenAI API instead, and contact admin."
            )
        return model

    def _load_tokenizer(
        self,
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        except:
            raise ValueError("Issue loading model. Use OpenAI API instead.")
        return tokenizer

    def local_model_api(
        self,
        input_dict: str,
    ):
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        input_text = tokenizer.apply_chat_template(
            input_dict, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output


if __name__ == "__main__":
    llama = LlamaInference(
        device=torch.device("cuda"),
    )
    print(llama.local_model_api("Was ist der name?"))
    pass
