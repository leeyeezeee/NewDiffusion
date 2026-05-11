import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from mas.llm.format import Message
from mas.llm.price import cost_count
from mas.llm.llm import LLM
from mas.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
async def achat(model_name:str, messages:list):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {authorization_key}'
    }

    # Normalize messages to OpenAI format
    norm_messages = []
    for m in messages:
        if hasattr(m, "to_dict"):
            norm_messages.append(m.to_dict())
        elif isinstance(m, dict):
            norm_messages.append(m)
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    
    data = {
        "model": model_name,
        "messages": norm_messages,
        "stream": False,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            response_data = await response.json()
            if 'choices' not in response_data:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"OpenAI API Error: {error_message}")
            prompt = "".join([m.get("content", "") for m in norm_messages])
            completion = response_data['choices'][0]['message']['content']
            cost_count(prompt, completion, model_name)
            return completion

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass