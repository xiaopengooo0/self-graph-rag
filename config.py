import logging
import os
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"]  ="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"
os.environ["HF_HOME"] = os.path.join(r"E:\dev\huggingface")


# 加载 .env 文件中的环境变量
load_dotenv()


logger = logging.getLogger(__name__)

@dataclass
class Neo4jConfig:
    """Neo4j数据库配置信息"""
    uri:str = os.getenv("NEO4J_URI", "localhost") # 数据库URI
    user:str = os.getenv("NEO4J_USER", "neo4j") # 数据库用户名
    password:str = os.getenv("NEO4J_PASSWORD", "") # 数据库密码
    driver:None ="" # Neo4j驱动
    database:str = os.getenv("NEO4J_DATABASE", "neo4j") # 数据库名称


@dataclass
class MilvusConfig:
    """Milvus数据库配置信息"""
    host: str = os.getenv("MILVUS_HOST", "localhost") # 数据库主机
    port: int = os.getenv("MILVUS_PORT", 19530)
    collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "cooking_knowledge")
    milvus_dimension: int = os.getenv("MILVUS_DIMENSION", 512)   # BGE-small-zh-v1.5的向量维度


@dataclass
class LLMConfig:
    """LLM配置信息"""
    model_name: str = os.getenv("LLM_MODEL_NAME","zai-org/GLM-4.6")
    api_key: str = os.getenv("LLM_API_KEY")
    api_base: str = os.getenv("LLM_BASE_URL")
    max_tokens: int = os.getenv("LLM_MAX_TOKENS", 2048) or 2048
    temperature: float = os.getenv("LLM_TEMPERATURE", 0.1) or 0.1
    top_k: int = os.getenv("LLM_TOP_K", 3) or 3


class GraphRAGConfig:
    """GraphRAG系统配置信息"""
    # Neo4j数据库配置
    neo4j_config: Neo4jConfig = Neo4jConfig()
    logger.debug(f"Neo4j数据库配置: {neo4j_config}")

    # Milvus数据库配置
    milvus_config: MilvusConfig = MilvusConfig()
    # LLM配置
    llm_config: LLMConfig = LLMConfig()


    # 向量模型配置
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5") or "BAAI/bge-small-zh-v1.5"


    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度


    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict:Dict[str, Any]) -> "GraphRAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# 初始化时加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_CONFIG = GraphRAGConfig()