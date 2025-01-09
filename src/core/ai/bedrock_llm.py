import json
import logging
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from crewai.llm import LLM

from config.config import LLM_CONFIG, AWS_BEDROCK_CONFIG

logger = logging.getLogger(__name__)

class BedrockLLM(LLM):
    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs
        )

        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_BEDROCK_CONFIG["region_name"],
            aws_access_key_id=AWS_BEDROCK_CONFIG["aws_access_key_id"],
            aws_secret_access_key=AWS_BEDROCK_CONFIG["aws_secret_access_key"],
            aws_session_token=AWS_BEDROCK_CONFIG.get("aws_session_token"),
            config=Config(retries={"max_attempts": 3, "mode": "standard"})
        )

    def call(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            # Prepare the request payload
            request_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {
                        "role": message["role"],
                        "content": [{"type": "text", "text": message["content"]}],
                    }
                    for message in messages
                ],
            }

            if self.stop:
                request_payload["stop_sequences"] = self.stop if isinstance(self.stop, list) else [self.stop]

            # Convert the request payload to JSON
            request = json.dumps(request_payload)

            # Invoke the model
            print(self.model   )
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model,
                body=request
            )

            # Decode the response body
            model_response = json.loads(response["body"].read())

            # Extract and return the response text
            return model_response["content"][0]["text"]

        except Exception as e:
            logger.error(f"Bedrock LLM call failed: {str(e)}")
            raise

    def supports_function_calling(self) -> bool:
        # Implement if Bedrock supports function calling
        return False

    def supports_stop_words(self) -> bool:
        # Bedrock supports stop sequences
        return True

    def get_context_window_size(self) -> int:
        # Return the context window size for the specific Bedrock model
        # You may need to adjust this based on the actual model being used
        return LLM_CONFIG.get("max_tokens", 8192)
