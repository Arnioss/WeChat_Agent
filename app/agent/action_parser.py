import ast
import re
from typing import Any, List, Optional, Tuple


class ActionParser:
    """功能：解析模型产出的 `<action>` 函数调用文本。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def parse(self, code_str: str) -> Tuple[str, List[Any]]:
        """功能：把 action 文本解析为函数名与位置参数列表。
        参数：
        - code_str：模型输出的 action 代码文本。
        返回值：
        - Tuple[str, List[Any]]：函数名与参数列表。
        异常：
        - ValueError：action 语法不合法或包含不支持参数时抛出。
        """
        code_str = self._sanitize_action_text(code_str)
        if self._looks_like_unclosed_string_call(code_str):
            raise ValueError("Action appears to contain an unclosed string literal or function call.")
        try:
            parsed = ast.parse(code_str, mode="eval")
        except SyntaxError as e:
            fallback = self._try_parse_common_malformed_action(code_str)
            if fallback is not None:
                return fallback
            raise ValueError(f"Invalid function call syntax: {e}") from e

        if not isinstance(parsed.body, ast.Call):
            raise ValueError("Action must be a function call")
        if not isinstance(parsed.body.func, ast.Name):
            raise ValueError("Only direct function names are allowed in action")
        if parsed.body.keywords:
            raise ValueError("Keyword arguments are not allowed in action")

        func_name = parsed.body.func.id
        args = [self._parse_arg(code_str, arg) for arg in parsed.body.args]
        return func_name, args

    def _parse_arg(self, code_str: str, arg_node: ast.AST) -> Any:
        """功能：解析单个参数 AST 节点为 Python 值。
        参数：
        - code_str：原始 action 文本。
        - arg_node：参数对应的 AST 节点。
        返回值：
        - Any：解析后的参数结果对象。
        异常：
        - ValueError：参数既非字面量也非受支持标识符时抛出。
        """
        try:
            return ast.literal_eval(arg_node)
        except (ValueError, SyntaxError):
            if isinstance(arg_node, ast.Name):
                return arg_node.id

            source = ast.get_source_segment(code_str, arg_node)
            if source is not None:
                source = source.strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./:-]*", source):
                    return source

            raise ValueError(
                f"Unsupported action argument: {ast.dump(arg_node, include_attributes=False)}"
            )

    @staticmethod
    def _try_parse_common_malformed_action(code_str: str) -> Optional[Tuple[str, List[Any]]]:
        # 兼容模型偶发输出：
        # rag_summarize(query: str) -> str: 测试环境信息怎么写示例
        """功能：兼容常见畸形 action 输出并尝试提取函数调用。
        参数：
        - code_str：原始 action 文本。
        返回值：
        - Optional[Tuple[str, List[Any]]]：成功时返回函数名与参数列表，失败返回 None。
        """
        text = (code_str or "").strip()
        if not text:
            return None

        # 只取第一行，避免流式截断带入无关后缀。
        first_line = text.splitlines()[0].strip()
        m = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\(\s*[^)]*\s*\)\s*(?:->\s*[^:]+)?\s*:\s*(.+)$",
            first_line,
        )
        if m:
            func_name = m.group(1)
            arg_text = (m.group(2) or "").strip()
            if not arg_text:
                return func_name, []
            # 去除可能附带的成对引号，避免双重引号。
            if (arg_text.startswith('"') and arg_text.endswith('"')) or (
                arg_text.startswith("'") and arg_text.endswith("'")
            ):
                arg_text = arg_text[1:-1]
            return func_name, [arg_text]
        return None

    @staticmethod
    def _sanitize_action_text(code_str: str) -> str:
        """功能：清洗 action 文本，移除标签与噪声片段。
        参数：
        - code_str：原始 action 文本。
        返回值：
        - str：清洗后的函数调用文本。
        """
        text = (code_str or "").strip()
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?thought>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"</?[^>]+>", " ", text)
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # 如果模型把完整回答串进 action，优先提取第一个函数调用片段
        m = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", text)
        if not m:
            return text
        return text[m.start() :].strip()

    @staticmethod
    def _looks_like_unclosed_string_call(text: str) -> bool:
        """功能：快速检测 action 是否存在未闭合括号或引号。
        参数：
        - text：待处理文本内容。
        返回值：
        - bool：疑似未闭合时返回 True。
        """
        value = (text or "").strip()
        if not value:
            return False
        # 快速判定：括号未闭合
        if value.count("(") > value.count(")"):
            return True
        # 引号未闭合（忽略转义）
        single_open = False
        double_open = False
        escaped = False
        for ch in value:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "'" and not double_open:
                single_open = not single_open
            elif ch == '"' and not single_open:
                double_open = not double_open
        return single_open or double_open
