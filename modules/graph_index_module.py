import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]  # 索引键列表
    value_content: str     # 详细描述内容
    entity_type: str       # 实体类型 (Recipe, Ingredient, CookingStep)
    metadata: Dict[str, Any]

@dataclass
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 多个索引键（可包含全局主题）
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体
    target_entity: str     # 目标实体
    metadata: Dict[str, Any]





class GraphIndexingModule:
    """
    图索引模块
    核心功能：
    1. 为实体创建键值对（名称作为唯一索引键）
    2. 为关系创建键值对（多个索引键，包含全局主题）
    3. 去重和优化图操作
    4. 支持增量更新
    """
    def __init__(self, config,llm_client):
        self.config = config
        self.llm_client = llm_client

        # 键值对存储
        self.entity_kv_store: Dict[str, EntityKeyValue] = {} # 实体键值对
        self.relation_kv_store: Dict[str, RelationKeyValue] = {} # 关系键值对

        #索引 key -> entity/relation IDs
        self.key_to_entities: Dict[str, List[str]] =  defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] =  defaultdict(list)


    def create_entity_key_values(self,
                                 recipes:List[Any],
                                 ingredients:List[Any],
                                 cooking_steps:List[Any]) -> Dict[str, EntityKeyValue]:
        """
        为实体创建键值对结构
        每个实体使用其名称作为唯一索引键
        """
        logger.info("开始创建实体键值对...")

        for recipe in recipes:
            entity_id = recipe.node_id
            entity_name = recipe.name or f"菜谱_{entity_id}"

            # 构建详细内容
            content_parts = [f"菜品名称：{entity_name}"]
            if hasattr(recipe, "properties"):
                props = recipe.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('category'):
                    content_parts.append(f"分类: {props['category']}")
                if props.get('cuisineType'):
                    content_parts.append(f"菜系: {props['cuisineType']}")
                if props.get('difficulty'):
                    content_parts.append(f"难度: {props['difficulty']}")
                if props.get('cookingTime'):
                    content_parts.append(f"制作时间: {props['cookingTime']}")

            # 创建键值对
            entity_kv = EntityKeyValue( entity_name= entity_name,
                                       index_keys = [entity_name],
                                       value_content="\n".join(content_parts),
                                       entity_type="Recipe",
                                       metadata={
                                           "node_id": entity_id,
                                           "properties":getattr(recipe, "properties", {})
                                       })

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)


            # 处理食材实体
        for ingredient in ingredients:
                entity_id = ingredient.node_id
                entity_name = ingredient.name or f"食材_{entity_id}"
                content_parts = [f"食材名称：{entity_name}"]

                if hasattr(ingredient, "properties"):
                    props = ingredient.properties
                    if props.get('category'):
                        content_parts.append(f"类别: {props['category']}")
                    if props.get('nutrition'):
                        content_parts.append(f"营养信息: {props['nutrition']}")
                    if props.get('storage'):
                        content_parts.append(f"储存方式: {props['storage']}")
                    entity_kv = EntityKeyValue( entity_name= entity_name,
                                               index_keys = [entity_name],
                                               value_content="\n".join(content_parts),
                                               entity_type="Ingredient",
                                               metadata={
                                                   "node_id": entity_id,
                                                   "properties":getattr(ingredient, "properties", {})
                                               })
                    self.entity_kv_store[entity_id] = entity_kv
                    self.key_to_entities[entity_name].append(entity_id)

            # 处理烹饪步骤
        for cooking_step in cooking_steps:
                entity_id = cooking_step.node_id
                entity_name = cooking_step.name or f"步骤_{entity_id}"
                content_parts = [f"步骤名称：{entity_name}"]

                if hasattr(cooking_step, "properties"):
                    props = cooking_step.properties
                    if props.get('description'):
                        content_parts.append(f"步骤描述: {props['description']}")
                    if props.get('order'):
                        content_parts.append(f"步骤顺序: {props['order']}")
                    if props.get('technique'):
                        content_parts.append(f"技巧: {props['technique']}")
                    if props.get('time'):
                        content_parts.append(f"时间: {props['time']}")


                entity_kv = EntityKeyValue( entity_name= entity_name,
                                           index_keys = [entity_name],
                                           value_content="\n".join(content_parts),
                                           entity_type="CookingStep",
                                           metadata={
                                               "node_id": entity_id,
                                               "properties": getattr(cooking_step, "properties", {})
                                           }
                )
                self.entity_kv_store[entity_id] = entity_kv
                self.key_to_entities[entity_name].append(entity_id)

                logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store


    def create_relation_key_value(self,
                                  relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """
        为关系创建键值对结构
        """
        logger.info("开始创建关系键值对...")

        for i , (source_id, target_id, relation_type) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"

            # 获取源节点和目标节点信息
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)

            if not source_entity or not target_entity:
                logger.warning(f"源节点或目标节点不存在，跳过关系 {relation_id}")
                continue

            content_parts = [
                f"关系类型：{relation_type}",
                f"源节点：{source_entity.entity_name}",
                f"目标节点：{target_entity.entity_name}"
            ]

            # 生成多个索引键
            index_keys = self._generate_relation_index_keys(
                source_entity,
                target_entity,
                relation_type
            )

            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content="\n".join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity= target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )

            self.relation_kv_store[relation_id] = relation_kv

            # 为每个索引建立映射
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store

    def _generate_relation_index_keys(self, source_entity, target_entity, relation_type) -> List[str]:
        """
        为关系生成多个索引键，包含全局主题
        """
        keys = [relation_type]

        # 根据关系类型和实体类型生成主题键
        if relation_type == "REQUIRES":
            # 菜谱-食材关系的主题键
            keys.extend( [
                "食材搭配",
                "烹饪原料",
                f"{source_entity.entity_name}_食材",
                target_entity.entity_name
            ])
        elif relation_type == "HAS_STEP" :
            # 菜谱-步骤关系的主题键
            keys.extend( [
                "制作步骤",
                "烹饪过程",
                f"{source_entity.entity_name}_步骤",
                "制作方法"
            ])
        elif relation_type == "BELONGS_TO_CATEGORY":
            # 分类关系的主题键
            keys.extend([
                "菜品分类",
                "美食类别",
                target_entity.entity_name
            ])

        # 使用LLm增强关系索引键
        if hasattr(self.config,"enable_llm_relation_keys",False):
            enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced_keys)


        return list(set(keys))

    def _llm_enhance_relation_keys(self, source_entity, target_entity, relation_type):
        """
                使用LLM增强关系索引键，生成全局主题
                """
        prompt = f"""
                分析以下实体关系，生成相关的主题关键词：

                源实体: {source_entity.entity_name} ({source_entity.entity_type})
                目标实体: {target_entity.entity_name} ({target_entity.entity_type})
                关系类型: {relation_type}

                请生成3-5个相关的主题关键词，用于索引和检索。
                返回JSON格式：{{"keywords": ["关键词1", "关键词2", "关键词3"]}}
                """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result.get("keywords", [])

        except Exception as e:
            logger.error(f"LLM增强关系索引键失败: {e}")
            return []