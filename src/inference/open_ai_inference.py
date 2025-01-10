from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from typing import List, Dict, Text

env_file = ".env.dev"
load_dotenv(env_file)

class OpenAIInference:
    def __init__(self):
        self.client = self.initialize_open_ai_client()
        pass
    
    def initialize_open_ai_client(
        self,
    ):
        try:
            client = OpenAI()
            return client
        except Exception as e:
            print(f"Following error occurred while trying to initialize Open-AI client: {e}.")
            raise
            
    def generate_response(
        self, 
        message: List[Dict] = None,
    ) -> Text:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Write a haiku about recursion in programming."
                    }
                ]
            )
            return response.choices[0].message.content
        except (RateLimitError, APIError) as e:
            print(f"error : {e} occured while trying to get response")
            raise
        except Exception as e:
            print(f"error : {e} occured while trying to get response")
            raise

if __name__=="__main__":
    inference = OpenAIInference()
    print(inference.generate_response())