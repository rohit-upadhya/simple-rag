import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np  # type: ignore
import os
import pickle

from typing import Optional, List, Dict, Any
from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore
from dotenv import load_dotenv  # type: ignore

from src.encoder import Encoder
from src.faiss import FaissDB
from src.inference.inference import Inference
from src.utils.input_loader import InputLoader
from src.utils.prompter import Prompter

env_file = ".env.dev"
load_dotenv(env_file)


class Rag:
    def __init__(
        self,
        input_file: Optional[str] = None,
        device: str = "cpu",
        load_encodings: bool = False,
        min_texts: int = 3,
        prompt_template_file: str = "resources/prompt_template.yaml",
        use_local: str = True,
    ):
        self.load_encodings = load_encodings
        self.input_file = input_file
        self.prompt_template_file = prompt_template_file
        self.use_local = use_local
        self.device = torch.device(device)
        self.encoder = Encoder(device=device)
        self.faiss = FaissDB()
        self.input_loader = InputLoader()
        self.ids = None
        self.min_texts = min_texts
        self.inference = Inference(device=self.device, use_local=self.use_local)
        self.load_input()

    def load_input(
        self,
    ):
        rag_dataset = load_dataset("neural-bridge/rag-dataset-12000")

        self.dataset = [item for item in rag_dataset["train"]]
        self.dataset = self.dataset[:100]
        self.ids = [idx for idx in range(0, len(self.dataset))]
        self.prompt_template = self.input_loader.load_file(
            file_name=self.prompt_template_file
        )

    def encode_dataset(self, datapoints, batch_size=16):
        num_datapoints = len(datapoints)
        all_encoded = []
        all_text = []
        for item in datapoints:
            all_text.append(item["context"])
        with tqdm(
            total=len(datapoints), desc=f"encoding...", unit="paragraph"
        ) as progress_bar:
            for start_idx in range(0, num_datapoints, batch_size):
                end_idx = min(start_idx + batch_size, num_datapoints)
                batch_paragraphs = all_text[start_idx:end_idx]
                encoded_batch = (
                    self.encoder.encode(batch_paragraphs).detach().cpu().numpy()
                )

                all_encoded.append(encoded_batch)

                progress_bar.update(end_idx - start_idx)
        all_encodings = np.vstack(all_encoded)

        return all_encodings

    def prompt_builder(
        self,
        context_text: str,
        query_text: str,
        prompt_template: dict[Any, Any],
    ) -> list[dict[str, str]]:
        return Prompter(
            context_text=context_text,
            query_text=query_text,
            prompt_template=prompt_template,
        ).build_chat_prompt()

    def build_index(self, all_encodings, ids):
        self.faiss.build_index(all_encodings, ids)

    def encode_query(
        self,
        query: str,
    ) -> np.array:

        return self.encoder.encode(query).detach().cpu().numpy()

    def retreiver(
        self,
        query_encoding,
        k: int = 3,
        threshold: float = 0.6,
    ) -> str:

        scores, match_list = self.faiss.perform_search_with_scores(query_encoding, k=k)

        top_score = max(scores)

        if top_score < threshold:
            return "No relevant documents found"

        match_list = [item for item in match_list if item != -1]
        match_list = match_list[: min(self.min_texts, len(match_list))]

        final_text = ""
        for item in match_list:
            final_text = f"{final_text}{self.dataset[item]}\n"
        return final_text

    def api_caller(self, message: list[dict]) -> str:
        response = self.inference.generate_results(input_dict=message)
        return response

    def chat_bot(self, query: str) -> str:
        encoded_data_path = "resources/encoded_paras.pickle"
        if os.path.isfile(encoded_data_path) and self.load_encodings:
            with open(encoded_data_path, "rb") as stream:
                encoded_data = pickle.load(stream)
        else:
            encoded_data = self.encode_dataset(self.dataset)
            with open(encoded_data_path, "wb") as stream:
                pickle.dump(encoded_data, stream)

        self.build_index(encoded_data, self.ids)
        query_encoding = self.encode_query(query=query)

        if self.encoder.model:
            del self.encoder.model

        final_match_text = self.retreiver(query_encoding)

        input_chat_message = self.prompt_builder(
            context_text=final_match_text,
            query_text=query,
            prompt_template=self.prompt_template,
        )

        response = self.api_caller(input_chat_message)
        return response


if __name__ == "__main__":
    rag = Rag(
        # device="cuda",
    )
    rag.chat_bot("What is the Berry Export Summary 2028 and what is its purpose?")
