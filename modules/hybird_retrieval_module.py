import logging
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from neo4j import GraphDatabase
from sqlalchemy.orm import relationship

from modules import GraphIndexingModule

logger = logging.getLogger(__name__)
class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. Round-robin轮询合并策略
    """

    def __init__(self,config,milvus_module,data_module,llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client

        self.driver = None
        self.bm25_retriever = None # BM25检索


        self.graph_indexing = GraphIndexingModule(config,llm_client)
        self.graph_indexed = False

    def initialize(self, chunks:List[Document]):
        """初始化"""
        logger.info("初始化混合检索模块...")

        # 连接neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_config.uri,
            auth=(self.config.neo4j_config.user, self.config.neo4j_config.password)
        )

        # 创建BM25检索
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")

        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引"""

        if self.graph_indexed :
            return

        logger.info("开始构建图索引...")
        try :
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_step = self.data_module.cooking_steps

            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_step)

            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_value(relationships)

            # 去重和优化图操作
            self.graph_indexing.deduplicate_entities_and_relations()
            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _extract_relationships_from_graph(self):
        """从Neo4j图中提取关系"""
        relationships = []

        try :
            with self.config.neo4j_config.driver.session() as session:
                query = """
                                MATCH (source)-[r]->(target)
                                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                                LIMIT 1000
                                """
                result = session.run(query)
                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))

        except Exception as e:
            logger.error(f"提取图关系失败: {e}")


        return  relationships