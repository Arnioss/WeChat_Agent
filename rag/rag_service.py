import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from rag.prompt_loader import load_rag_prompt
from rag.vector_store import VectorStoreService


class RagSummarizeService:
    """功能：封装本地知识库检索与大模型总结流程。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, project_directory: str):
        """功能：初始化 RAG 服务依赖，包括向量检索、提示模板与对话模型客户端。
        参数：
        - project_directory：项目根目录路径。
        返回值：
        - 无。若缺少模型或网关配置会立即抛出异常，避免在检索流程中延迟失败。
        """
        load_dotenv()
        self.project_directory = str(Path(project_directory).resolve())
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompt(self.project_directory)
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raw_keys = (os.getenv("OPENROUTER_API_KEYS") or "").strip()
            api_key = raw_keys.split(",")[0].strip() if raw_keys else ""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = os.getenv("RAG_CHAT_MODEL") or os.getenv("OPENROUTER_MODEL")
        if not self.model:
            raise ValueError("缺少环境变量 RAG_CHAT_MODEL 或 OPENROUTER_MODEL，请在 .env 文件中设置。")

    def rag_summarize(self, query: str) -> str:
        """功能：基于本地向量库检索结果生成回答文本。
        参数：
        - query：用户输入的问题文本。
        返回值：
        - 可直接返回给用户的回答文本或提示信息。
        """
        if not query.strip():
            return "请输入你要查询的知识问题。"
        self.vector_store.build_or_update_index()
        docs = self.vector_store.retrieve(query)
        if not docs:
            return "知识库暂无足够信息。请补充资料后重试。"

        context = []
        for i, d in enumerate(docs, start=1):
            source = d.get("source", "")
            content = (d.get("content", "") or "").strip()
            if not content:
                continue
            context.append(f"【参考资料{i}】\n来源: {source}\n内容: {content}")

        prompt = self.prompt_text.format(query=query, context="\n\n".join(context))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

