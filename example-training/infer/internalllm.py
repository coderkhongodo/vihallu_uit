import json
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass, field

import httpx


REASONING_BUDGET = 8092


@dataclass
class LLMSamplingSettings:
    """Configuration settings for LLM text generation."""
    temperature: float = 0.1
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    n_predict: int = -1
    n_keep: int = 0
    stream: bool = False
    additional_stop_sequences: List[str] = field(default_factory=list)
    tfs_z: float = 1.0
    typical_p: float = 1.0
    # repetition_penalty: Optional[float] = None
    repeat_last_n: int = -1
    penalize_nl: bool = False
    presence_penalty: float = 1.1
    # frequency_penalty: float = 1.5
    penalty_prompt: Optional[Union[str, List[int]]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    cache_prompt: bool = True
    seed: int = -1
    ignore_eos: bool = False
    samplers: Optional[List[str]] = None
    response_format: Optional[str] = None

    def get_additional_stop_sequences(self) -> List[str]:
        """Get the list of additional stop sequences."""
        return self.additional_stop_sequences

    def add_additional_stop_sequences(self, sequences: List[str]) -> None:
        """Add new stop sequences to the existing list.
        
        Args:
            sequences: List of stop sequences to add
        """
        self.additional_stop_sequences.extend(sequences)

    def is_streaming(self) -> bool:
        """Check if streaming is enabled."""
        return self.stream

    def as_dict(self) -> Dict[str, Any]:
        """Convert the settings to a dictionary.
        
        Returns:
            Dictionary representation of the settings
        """
        return self.__dict__


class LLMServerProvider:
    """Provider class for interacting with LLM server endpoints."""

    def __init__(self, server_address: str):
        """Initialize the LLM server provider.
        
        Args:
            server_address: Base URL of the LLM server
            
        Raises:
            ValueError: If server_address is empty
        """
        if not server_address:
            raise ValueError("Server address cannot be empty.")

        self.server_address = server_address
        self.server_chat_completion_endpoint = (
            self.server_address + "/v1/chat/completions"
        )

    @staticmethod
    def get_provider_default_settings() -> LLMSamplingSettings:
        return LLMSamplingSettings()

    async def create_chat_completion(
        self,
        session: httpx.AsyncClient,
        messages: List[Dict[str, Any]],
        settings: Dict[str, Any],
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        thinking_mode: bool = False,
        response_format: Optional[str] = None,
    ):
        """Create a chat completion request to the LLM server.

        Args:
            session: httpx client session
            messages: List of message dictionaries with 'role' and 'content' keys
            settings: Generation settings
            api_key: Optional API key for authentication
            model: Model name to use
            thinking_mode: Whether to enable thinking mode
            
        Returns:
            Tuple of response, reasoning_content
            
        Raises:
            aiohttp.ClientError: If the request fails
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages_dict = [
            {
                "role": message["role"],
                "content": message["content"]
            }
            for message in messages
        ]

        data = deepcopy(settings)
        data["model"] = model
        data["messages"] = messages_dict
        data["stream"] = True

        if response_format:
            data["response_format"] = {"type": response_format}

        if thinking_mode:
            data['chat_template_kwargs'] = {"enable_thinking": True}
        else:
            data['chat_template_kwargs'] = {"enable_thinking": False}

        response, reasoning_content = await self.call_chat(
            data=data,
            headers=headers,
            session=session,
            reasoning_budget=REASONING_BUDGET
        )
        if thinking_mode and reasoning_content and not response:
            print("EXCEED REASONING BUDGET, RE-TRYING, REASONING CONTENT:", reasoning_content)
            data['chat_template_kwargs'] = {"enable_thinking": False}
            data['messages'].append({"role": "assistant", "content": f"<think>\n{reasoning_content}\n</think>\n\n"})
            response, reasoning_content = await self.call_chat(
                data=data,
                headers=headers,
                session=session,
                reasoning_budget=REASONING_BUDGET
            )
        return response, reasoning_content

    async def call_chat(
        self,
        data: Dict[str, Any],
        headers: Dict[str, str],
        session: httpx.AsyncClient,
        reasoning_budget: int = REASONING_BUDGET,
    ):
        data = self.prepare_generation_settings(data)
        response = ""
        reasoning_content = ""
        reasoning_tokens = 0
        async with session.stream(
                "POST",
                url=self.server_chat_completion_endpoint,
                headers=headers,
                json=data,
                timeout=None
        ) as result:
            async for chunk in result.aiter_bytes():
                try:
                    chunk_data = chunk.decode("utf-8").rstrip("\x00")
                    chunk_data = chunk_data.replace("data: ", "")
                    chunk_json = json.loads(chunk_data)
                    if result.status_code != 200:
                        continue
                    delta = chunk_json["choices"][0]["delta"]
                    if "content" in delta:
                        response += delta["content"]
                    if "reasoning_content" in delta:
                        reasoning_content += delta["reasoning_content"]
                        reasoning_tokens += 1
                except json.decoder.JSONDecodeError:
                    if '\n\n' in chunk_data:
                        data_chunks = chunk_data.split('\n\n')
                        for c in data_chunks:
                            if c.strip():
                                try:
                                    c_json = json.loads(c)
                                    if result.status_code != 200:
                                        continue
                                    delta = c_json["choices"][0]["delta"]
                                    if "content" in delta:
                                        response += delta["content"]
                                    if "reasoning_content" in delta:
                                        reasoning_content += delta["reasoning_content"]
                                        reasoning_tokens += 1
                                except json.decoder.JSONDecodeError:
                                    pass
                if reasoning_tokens >= reasoning_budget and not response:
                    break
        return response, reasoning_content

    @staticmethod
    def prepare_generation_settings(settings_dictionary: dict) -> dict:
        """
        Prepare generation settings for the LLM API call.
        
        Args:
            settings_dictionary: Dictionary of settings
            
        Returns:
            Modified settings dictionary ready for API call
        """
        settings_dictionary["mirostat"] = settings_dictionary.pop("mirostat_mode", 0)
        if "additional_stop_sequences" in settings_dictionary:
            settings_dictionary["stop"] = settings_dictionary.pop("additional_stop_sequences")
        if "samplers" in settings_dictionary:
            del settings_dictionary["samplers"]
        
        return {k: v for k, v in settings_dictionary.items() if v is not None}
    