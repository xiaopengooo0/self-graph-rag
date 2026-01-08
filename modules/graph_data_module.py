import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from config import Neo4jConfig

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id:str # 节点ID
    labels:List[str] # 节点标签
    name:str  # 节点名称
    properties:Dict[str, Any] # 节点属性

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id:str # 开始节点ID
    end_node_id:str # 结束节点ID
    relation_type:str # 关系类型
    properties:Dict[str, Any] # 关系属性






class GraphDataModule:
    def __init__(self,config= Neo4jConfig):
        self.neo4j_config = config

        self.documents: List[Document] = [] # 文档列表
        self.chunks: List[Document] = [] # 分块列表
        self.recipes: List[GraphNode] = [] # 菜谱列表
        self.ingredients: List[GraphNode] = [] # 食材列表
        self.cooking_steps: List[GraphNode] = [] # 烹饪步骤列表

        self._connect()

    def _connect(self):
        """建立Neo4j连接"""

        try:
            self.neo4j_config.driver = GraphDatabase.driver(
                self.neo4j_config.uri,
                auth=(self.neo4j_config.user, self.neo4j_config.password),
                database=self.neo4j_config.database
            )

            logger.info(f"已连接到Neo4j数据库: {self.neo4j_config.uri}")

            # 测试链接
            with self.neo4j_config.driver.session() as session:
                result = session.run("RETURN 1 as test")
                _result = result.single() # 获取结果
                if _result:
                    logger.info("Neo4j连接测试成功")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise


    def close(self):
        """关闭Neo4j连接"""
        if hasattr(self.neo4j_config, 'driver') and self.neo4j_config.driver:
            self.neo4j_config.driver.close()
            logger.info("已关闭Neo4j连接")
