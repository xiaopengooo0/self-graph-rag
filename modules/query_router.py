import logging

logger = logging.getLogger(__name__)


class IntelligentQueryRouter:
    """
    智能查询路由器

    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """
    def __init__(self,config,llm_client,traditional_retrieval,graph_retrieval):
        self.config = config
        self.llm_client = llm_client
        self.traditional_retrieval = traditional_retrieval
        self.graph_retrieval = graph_retrieval

        self.router_status = {
            "traditional_count":0,
            "graph_retrieval_count":0,
            "combined_count":0,
            "total_query":0
        }