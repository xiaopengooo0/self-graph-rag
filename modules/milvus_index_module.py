import logging

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

from config import MilvusConfig

logger = logging.getLogger(__name__)

class MilvusIndexModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self,config:MilvusConfig, embedding_model_name:str="BGE-small-zh-v1.5"):
        self.config = config
        self.embedding_model_name = embedding_model_name

        self.client = None

        self.embeddings =  None
        self.collection_created = False

        self._init_milvus()
        self._init_embeddings()

    def _init_milvus(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(host=self.config.host, port=self.config.port)
            logger.info(f"已连接到Milvus服务: {self.config.host}:{self.config.port}")

            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _init_embeddings(self):
        """初始化向量模型"""
        logger.info(f"正在初始化嵌入模型: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name,
                                                model_kwargs={"device": "cpu"}, # 使用CPU
                                                encode_kwargs={"normalize_embeddings": False} # 不进行归一化
                                                )

        logger.info(f"嵌入模型初始化完成")



    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def __del__(self):
        self.close()