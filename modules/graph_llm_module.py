import logging

from config import LLMConfig
from langchain_openai import ChatOpenAI
logger = logging.getLogger(__name__)

class LLMModule:
    """LLM模块 - 负责LLM模型调用"""
    def __init__(self,config:LLMConfig):
        self.config = config
        if not self.config.api_key:
            raise ValueError("LLM API KEY不能为空")

        self.client  = ChatOpenAI(api_key=self.config.api_key,
                                  model_name=self.config.model_name,
                                  base_url = self.config.api_base,
                                  temperature=self.config.temperature or 0.1,
                                  max_tokens=self.config.max_tokens or 2048,
                                  # top_k=self.config.top_k or 3
                                  )
        logger.info(f"已初始化LLM模型: {self.config.model_name}")
