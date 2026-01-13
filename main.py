import logging

from dotenv import load_dotenv

from config import GraphRAGConfig, DEFAULT_CONFIG
from modules import GraphDataModule, LLMModule, MilvusIndexModule, HybridRetrievalModule, GraphRAGRetrievalModule, \
    IntelligentQueryRouter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class AdvanceGraphRAGSystem:
    """
    图RAG系统

    核心特性：
    1. 智能路由：自动选择最适合的检索策略
    2. 双引擎检索：传统混合检索 + 图RAG检索
    3. 图结构推理：多跳遍历、子图提取、关系推理
    4. 查询复杂度分析：深度理解用户意图
    5. 自适应学习：基于反馈优化系统性能
    """
    def __init__(self, config:GraphRAGConfig = DEFAULT_CONFIG):
        self.config = config

        # 核心模块
        self.data_module = None
        self.index_module = None
        self.llm_module = None

        # 检索引擎
        self.traditional_retrieval = None
        self.graph_retrieval = None
        self.query_router = None

        self.system_status = False


    def init_system(self):
        """
        初始化系统
        """
        logger.info("正在启动高级图RAG系统...")
        # 1. 数据准备模块
        print("1.初始化数据准备模块...")
        self.data_module = GraphDataModule(self.config.neo4j_config)
        # 2. 向量索引模块
        print("2.初始化索引模块...")
        self.index_module = MilvusIndexModule(self.config.milvus_config,self.config.embedding_model_name)

        # 3. 生成模块
        print("3.初始化LLM模块...")
        self.llm_module = LLMModule(self.config.llm_config)
        # 4. 传统混合检索模块
        print("4.初始化传统混合检索...")
        self.traditional_retrieval = HybridRetrievalModule(config = self.config,
                                                           milvus_module=self.index_module,
                                                           data_module= self.data_module,
                                                           llm_client= self.llm_module.client)

        #5.图RAG索引模块
        print("5.初始化图RAG索引模块...")
        self.graph_retrieval = GraphRAGRetrievalModule(config = self.config,
                                                           llm_client= self.llm_module.client)

        #6.智能查询路由
        print("6.初始化智能查询路由...")
        self.query_router = IntelligentQueryRouter(config = self.config,
                                                           llm_client= self.llm_module.client,
                                                           graph_retrieval= self.graph_retrieval,
                                                           traditional_retrieval= self.traditional_retrieval)




if __name__ == '__main__':
    system = AdvanceGraphRAGSystem()
    system.init_system()