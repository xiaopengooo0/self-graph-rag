
class GraphRAGRetrievalModule:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """

    def __init__(self,config,llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None


        # 图结构缓存
        self.entity_cache = {} # 实体缓存
        self.relation_cache = {} # 关系缓存
        self.subgraph_cache = {} # 子图缓存