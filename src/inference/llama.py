from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class LlamaInference:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model_name_or_path: str = "meta-llama/Llama-3.2-1B",
        load_quantized: bool = True,
        quantization_bits: int = 4,
    ):
        self.model_name_or_path = model_name_or_path
        self.quantization_bits = quantization_bits
        self.load_quantized = load_quantized
        self.device = device

    def _quantization_config(
        self,
    ):
        load_in_4bit = False
        load_in_8bit = False
        if self.quantization_bits == 4:
            load_in_4bit = True
        else:
            load_in_8bit = True
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_use_double_quant=load_in_4bit,
            bnb_4bit_quant_type="nf4",
        )

        return bnb_config

    def _load_model(
        self,
        model_name_or_path: str = None,
    ):
        model_config = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        try:
            if model_name_or_path is None:
                model_name_or_path = self.model_name_or_path
            if self.load_quantized:
                quantization_config = self._quantization_config()
                model_config["quantization_config"] = quantization_config
            model_config["pretrained_model_name_or_path"] = model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(**model_config)
        except:
            raise ValueError(
                "Issue loading local model. Use OpenAI API instead, and contact admin."
            )
        return model

    def _load_tokenizer(
        self,
        model_name_or_path: str = None,
    ):
        try:
            if model_name_or_path is not None:
                model_name_or_path = self.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained()
        except:
            raise ValueError("Issue loading model. Use OpenAI API instead.")
        return tokenizer

    def local_model_api(
        self,
        input_dict: str,
    ):
        model = self._load_model(model_name_or_path=self.model_name_or_path)
        tokenizer = self._load_tokenizer(model_name_or_path=self.model_name_or_path)
        input_text = tokenizer.apply_chat_template(
            input_dict, tokenize=False, add_generation_prompt=False
        )
        print(input_text)
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        return outputs


if __name__ == "__main__":

    pass
