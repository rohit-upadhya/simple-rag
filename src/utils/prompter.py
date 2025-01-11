

from typing import Text, Dict, Any, List

from src.utils.input_loader import InputLoader

class Prompter:
    def __init__(
        self,
        context_text: Text,
        query_text: Text,
        prompt_template: Dict[Any, Any]
    ):
        self.prompt_template = prompt_template
        self.query_text = query_text
        self.context_text = context_text
        pass
    
    def build_chat_prompt(
        self,
    ) -> List[Dict[Text, Text]]:
        final_prompt = []
        if "system_prompt" in self.prompt_template:
            content = self.prompt_template.get("system_prompt", "").format(context_text=self.context_text)
            final_prompt.append(
                {
                    "role": "system",
                    "content": content
                }
            )
        
        if self.query_text is not None:
            final_prompt.append(
                {
                    "role": "user",
                    "content": self.query_text
                }
            )
        else:
            print("No text provided. Please provide a query and try again.")
            raise
        return final_prompt