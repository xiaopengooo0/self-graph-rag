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

        try:
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

            print("✅ 高级图RAG系统初始化完成！")
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def build_knowledge_base(self):
        """构建知识库（如果需要）"""
        print("正在构建知识库...")

        if self.index_module.has_collection():
            print("知识库已存在，尝试加载...")
            if self.index_module.load_collection():
                print("📚 知识库加载成功！")

                print("加载图数据以支持图检索...")
                self.data_module.load_graph_data()
                print("📗 构建菜谱文档")
                self.data_module.build_recipe_documents()
                print("进行文档分块...")
                chunks = self.data_module.chunk_documents(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                print("构建索引...")
                self._init_retrievers(chunks)
                return
            else:
                print("❌ 知识库加载失败，开始重建...")
        print("未找到已存在的集合，开始构建新的知识库...")

        # 从Neo4j加载图数据
        print("从Neo4j加载图数据...")
        self.data_module.load_graph_data()

        # 构建菜谱文档
        print("构建菜谱文档...")
        self.data_module.build_recipe_documents ()

        # 文档分块

        print("文档分块...")
        chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        # 构建Milvus 向量索引
        print("构建Milvus 向量索引...")
        if not self.index_module.build_vector_index(chunks):
            print("构建Milvus 向量索引失败")
        # 初始化检索器
        self._initialize_retrievers(chunks)

        # 显示统计信息
        self._show_knowledge_base_stats()

        print("✅ 知识库构建完成！")
    def _init_retrievers(self, chunks):

        print("初始化检索引擎...")

        if chunks is None:
            chunks  = self.data_module.chunks or []

        # 初始化传统检索
        self.traditional_retrieval.initialize(chunks)
        # 初始化图检索
        self.graph_retrieval.initialize()

        self.system_status = True

        print("✅ 检索引擎初始化完成！")

    def _initialize_retrievers(self, chunks):
        """初始化检索器"""
        print("初始化检索引擎...")

        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []

        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)

        # 初始化图RAG检索器
        self.graph_retrieval.initialize()

        self.system_ready = True
        print("✅ 检索引擎初始化完成！")

    def _show_knowledge_base_stats(self):
        """显示知识库统计信息"""
        print(f"\n知识库统计:")

        # 数据统计
        stats = self.data_module.get_statistics()
        print(f"   菜谱数量: {stats.get('total_recipes', 0)}")
        print(f"   食材数量: {stats.get('total_ingredients', 0)}")
        print(f"   烹饪步骤: {stats.get('total_cooking_steps', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")

        # Milvus统计
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")

        # 图RAG统计
        route_stats = self.query_router.get_route_statistics()
        print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")

        if stats.get('categories'):
            categories = list(stats['categories'].keys())[:10]
            print(f"   🏷️ 主要分类: {', '.join(categories)}")


if __name__ == '__main__':
    print("GraphRAG系统启动中...")
    # 创建高级图RAG系统实例
    system = AdvanceGraphRAGSystem()
    # 初始化系统
    system.init_system()

    #  构建知识库
    system.build_knowledge_base()