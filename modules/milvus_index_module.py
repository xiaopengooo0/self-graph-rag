import logging
import time
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

from config import MilvusConfig

logger = logging.getLogger(__name__)


def _safe_truncate(text: str, max_length: int):
    """
    安全截取字符串，处理None值

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        截取后的字符串
    """
    if text is None:
        return ""
    return str(text)[:max_length]


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

    def has_collection(self) -> bool:
        """
        检查集合是否存在

        Returns:
            集合是否存在
        """
        try:
            return self.client.has_collection(self.config.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False

    def load_collection(self) -> bool:
        """
        加载集合到内存

        Returns:
            是否加载成功
        """
        try:
            if not self.client.has_collection(self.config.collection_name):
                logger.error(f"集合 {self.config.collection_name} 不存在")
                return False

            self.client.load_collection(self.config.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.config.collection_name} 已加载到内存")
            return True

        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """构建向量索引"""
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档列表为空")
        # 1.创建集合
        if not self.create_collection(force_recreate=True):
            return False
        # 2.准备数据
        logger.info("正在生成向量embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents( texts)

        # 3.向集合中插入数据
        entities = []
        for i ,(chunk, vector) in enumerate(zip(chunks, vectors)):
            entity = {
                "id": _safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                "vector": vector,
                "text": _safe_truncate(chunk.page_content, 15000),
                "node_id": _safe_truncate(chunk.metadata.get("node_id", ""), 100),
                "recipe_name": _safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                "node_type": _safe_truncate(chunk.metadata.get("node_type", ""), 100),
                "category": _safe_truncate(chunk.metadata.get("category", ""), 100),
                "cuisine_type": _safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                "difficulty": int(chunk.metadata.get("difficulty", 0)),
                "doc_type": _safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                "chunk_id": _safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                "parent_id": _safe_truncate(chunk.metadata.get("parent_id", ""), 100)
            }
            entities.append(entity)

        # 4.批量插入数据
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            self.client.insert(
                collection_name=self.config.collection_name,
                data=batch,
                timeout=60
            )
            logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")

        # 5.创建索引
        if not self.create_index():
            return  False

        #6. 加载索引到内存
        self.client.load_collection(self.config.collection_name)
        logger.info("集合已加载到内存")

        # 7. 等待索引构建完成
        logger.info("等待索引构建完成...")
        time.sleep(2)

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return True

    def create_collection(self, force_recreate: bool = False):
        """创建Milvus集合 force_recreate: 是否强制重新创建集合"""
        try:
            if self.client.has_collection(self.config.collection_name):
                if force_recreate:
                    self.client.drop_collection(self.config.collection_name)
                else:
                    logger.info(f"集合 {self.config.collection_name} 已经存在")
                    self.collection_created = True
                    return True
            # 创建集合
            schema = self._create_collection_schema()
            self.client.create_collection(self.config.collection_name,
                                          schema = schema,
                                          metric_type="COSINE",  # 使用余弦相似度
                                          consistency_level="Strong",
                                          dimension=self.config.milvus_dimension
                                          )
            logger.info(f"创建集合 {self.config.collection_name} 成功")
            self.collection_created = True
            return True
        except Exception as e:
            logger.error(f"创建Milvus集合失败: {e}")
            return False


    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_collection_schema(self)->CollectionSchema:
        """
        创建集合模式

        Returns:
            集合模式对象
        """
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.milvus_dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]

        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合"
        )

        return schema

    def create_index(self):
        """创建向量索引"""

        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")

            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()

            # 添加向量字段
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )
            self.client.create_index(
                collection_name=self.config.collection_name,
                index_params=index_params
            )

            logger.info("向量索引创建成功")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}

            stats = self.client.get_collection_stats(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0),
                "stats": stats
            }

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
