from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from app.skills.models import SkillMatch, SkillMetadata


class SkillRetrievalStrategy(Protocol):
    """功能：定义技能检索策略协议，约束检索接口形态。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def retrieve(self, query: str, metadata: Iterable[SkillMetadata], *, limit: int) -> tuple[SkillMatch, ...]:
        """功能：按查询内容检索最相关的技能或文档。
        参数：
        - query：用户输入的问题文本。
        - metadata：技能元数据列表或对象。
        - limit：返回结果数量上限。
        返回值：
        - 无。
        """
        ...


@dataclass(frozen=True)
class RuleBasedSkillRetrievalStrategy:
    """功能：基于规则和关键词匹配计算技能候选分数。
    参数：
    - 无。
    返回值：
    - 无。
    """
    min_score: float = 1.0

    def retrieve(self, query: str, metadata: Iterable[SkillMetadata], *, limit: int) -> tuple[SkillMatch, ...]:
        """功能：按查询内容检索最相关的技能或文档。
        参数：
        - query：用户输入的问题文本。
        - metadata：技能元数据列表或对象。
        - limit：返回结果数量上限。
        返回值：
        - tuple[SkillMatch, ...]：按分数降序返回的技能匹配结果，数量不超过 `limit`。
        """
        normalized_query = self._normalize_text(query)
        query_terms = self._tokenize(normalized_query)
        matches: list[SkillMatch] = []
        for item in metadata:
            if item.disable_model_invocation:
                continue
            score, reasons, matched_terms = self._score_item(item, normalized_query, query_terms)
            if score < self.min_score:
                continue
            matches.append(
                SkillMatch(
                    skill_name=item.name,
                    score=score,
                    source="rule-based",
                    match_reasons=tuple(reasons),
                    matched_terms=tuple(matched_terms),
                    allow_auto_activation=not item.disable_model_invocation,
                    decision="candidate",
                )
            )
        matches.sort(key=lambda item: (-item.score, item.skill_name))
        return tuple(matches[: max(0, limit)])

    def _score_item(
        self,
        item: SkillMetadata,
        normalized_query: str,
        query_terms: tuple[str, ...],
    ) -> tuple[float, list[str], list[str]]:
        """功能：为单个技能计算匹配得分并给出命中原因。
        参数：
        - item：待评估的技能元数据。
        - normalized_query：归一化后的查询文本。
        - query_terms：查询拆分后的词项集合。
        返回值：
        - tuple[float, list[str], list[str]]：依次为总分、去重后的命中原因列表、命中的查询词列表。
        """
        score = 0.0
        reasons: list[str] = []
        matched_terms: list[str] = []

        name_text = self._normalize_text(item.name.replace("-", " "))
        alias_texts = tuple(self._normalize_text(alias) for alias in item.aliases)
        keyword_texts = tuple(self._normalize_text(keyword) for keyword in item.keywords)
        description_text = self._normalize_text(item.description)
        tags_text = tuple(self._normalize_text(tag) for tag in item.tags)
        examples_text = tuple(self._normalize_text(example) for example in item.examples)

        phrase_fields = (
            ("alias", alias_texts, 8.0),
            ("keyword", keyword_texts, 7.0),
            ("example", examples_text, 4.0),
            ("tag", tags_text, 3.0),
        )

        if normalized_query and normalized_query == name_text:
            score += 10.0
            reasons.append(f"query exactly matches skill name '{item.name}'")

        if normalized_query and normalized_query in alias_texts:
            score += 8.0
            reasons.append(f"query exactly matches alias of '{item.name}'")

        if normalized_query and normalized_query in description_text:
            score += 4.0
            reasons.append("query text appears in description")

        for field_name, values, weight in phrase_fields:
            for value in values:
                if not value or len(value) < 2:
                    continue
                if value in normalized_query:
                    score += weight
                    reasons.append(f"{field_name} phrase '{value}' appears in query")
                    if value not in matched_terms:
                        matched_terms.append(value)

        domain_score, domain_reasons, domain_terms = self._domain_boost(item.name, normalized_query)
        if domain_score:
            score += domain_score
            reasons.extend(domain_reasons)
            for term in domain_terms:
                if term not in matched_terms:
                    matched_terms.append(term)

        for term in query_terms:
            if len(term) < 2:
                continue
            if self._is_weak_cjk_term(term):
                term_weight_scale = 0.35
            else:
                term_weight_scale = 1.0
            term_hit = False
            if term in name_text:
                score += 4.0 * term_weight_scale
                reasons.append(f"term '{term}' matched skill name")
                term_hit = True
            if any(term in alias for alias in alias_texts):
                score += 3.5 * term_weight_scale
                reasons.append(f"term '{term}' matched aliases")
                term_hit = True
            if any(term in keyword for keyword in keyword_texts):
                score += 3.0 * term_weight_scale
                reasons.append(f"term '{term}' matched keywords")
                term_hit = True
            if any(term in tag for tag in tags_text):
                score += 2.5 * term_weight_scale
                reasons.append(f"term '{term}' matched tags")
                term_hit = True
            if term in description_text:
                score += 1.5 * term_weight_scale
                reasons.append(f"term '{term}' matched description")
                term_hit = True
            if any(term in example for example in examples_text):
                score += 1.0 * term_weight_scale
                reasons.append(f"term '{term}' matched examples")
                term_hit = True
            if term_hit and term not in matched_terms:
                matched_terms.append(term)

        if matched_terms:
            score += min(float(item.priority), 5.0) * 0.1

        deduped_reasons = []
        seen = set()
        for reason in reasons:
            if reason in seen:
                continue
            seen.add(reason)
            deduped_reasons.append(reason)
        return score, deduped_reasons, matched_terms

    @staticmethod
    def _normalize_text(text: str) -> str:
        """功能：归一化文本，统一空白与大小写格式。
        参数：
        - text：待处理文本内容。
        返回值：
        - str：去首尾空白、压缩连续空白并转为小写后的文本。
        """
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        """功能：把查询文本切分为可用于匹配的词项集合。
        参数：
        - text：待处理文本内容。
        返回值：
        - tuple[str, ...]：去重后的英文词项与中文 n-gram 词项元组。
        """
        if not text:
            return ()
        latin_terms = re.findall(r"[a-z0-9][a-z0-9_-]*", text)
        cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        merged = []
        for term in latin_terms:
            if term not in merged:
                merged.append(term)
        for sequence in cjk_sequences:
            for term in RuleBasedSkillRetrievalStrategy._expand_cjk_sequence(sequence):
                if term not in merged:
                    merged.append(term)
        return tuple(merged)

    @staticmethod
    def _is_weak_cjk_term(term: str) -> bool:
        """功能：判断中文词项是否为泛化弱信号，匹配时应降低权重。
        参数：
        - term：待判断的词项。
        返回值：
        - bool：命中弱词表或为两字 CJK 词时返回 True。
        """
        weak_terms = {
            "版本",
            "环境",
            "测试",
            "文档",
            "流程",
            "配置",
            "平台",
            "查询",
            "回答",
        }
        return term in weak_terms or (len(term) == 2 and re.fullmatch(r"[\u4e00-\u9fff]{2}", term) is not None)

    @staticmethod
    def _domain_boost(skill_name: str, normalized_query: str) -> tuple[float, list[str], list[str]]:
        """功能：按技能领域词典为查询中的领域信号追加加分。
        参数：
        - skill_name：技能名称，用于选择对应领域词表。
        - normalized_query：归一化后的查询文本。
        返回值：
        - tuple[float, list[str], list[str]]：依次为加分、命中原因列表、命中的领域词列表。
        """
        groups = {
            "knowledge-rag-answer": (
                ("知识库", 8.0),
                ("按文档", 7.0),
                ("基于文档", 7.0),
                ("内部资料", 7.0),
                ("参考资料", 6.0),
                ("规范", 5.0),
                ("流程文档", 5.0),
            ),
            "testcase-generator": (
                ("prd", 8.0),
                ("测试用例", 9.0),
                ("测试点", 7.0),
                ("用例设计", 8.0),
                ("边界值", 6.0),
                ("异常场景", 6.0),
                ("权限测试", 6.0),
            ),
        }
        score = 0.0
        reasons: list[str] = []
        terms: list[str] = []
        for term, weight in groups.get(skill_name, ()):
            if term in normalized_query:
                score += weight
                reasons.append(f"domain signal '{term}' matched {skill_name}")
                terms.append(term)
        return score, reasons, terms

    @staticmethod
    def _expand_cjk_sequence(sequence: str) -> tuple[str, ...]:
        """功能：把连续中文片段扩展为 2/3 字词项，提升召回率。
        参数：
        - sequence：连续 CJK 字符序列。
        返回值：
        - tuple[str, ...]：去重后的中文词项元组。
        """
        terms = []
        if len(sequence) >= 2:
            terms.append(sequence)
            for idx in range(len(sequence) - 1):
                terms.append(sequence[idx : idx + 2])
        if len(sequence) >= 3:
            for idx in range(len(sequence) - 2):
                terms.append(sequence[idx : idx + 3])
        deduped = []
        for term in terms:
            if term not in deduped:
                deduped.append(term)
        return tuple(deduped)
