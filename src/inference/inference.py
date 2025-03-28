from src.inference.open_ai_inference import OpenAIInference
from src.inference.llama import LlamaInference
import torch


class Inference:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        use_local: bool = True,
    ):
        self.device = device
        self.use_local = use_local

    def generate_results(
        self,
        input_dict: list[dict],
    ):
        if self.use_local:
            generation_method = self._generate_local
        else:
            generation_method = self._generate_api

        return generation_method(input_dict)

    def _generate_local(
        self,
        input_dict: list[dict],
    ):
        local_inference = LlamaInference(
            model_name_or_path="meta-llama/Llama-3.2-1B",
            load_quantized=True,
            device=self.device,
        )
        desired_output = local_inference.local_model_api(input_dict=input_dict)
        return desired_output

    def _generate_api(
        self,
        input_dict: list[dict],
    ):
        self.open_ai_inference = OpenAIInference()
        response = self.open_ai_inference.generate_response(message=input_dict)
        return response
