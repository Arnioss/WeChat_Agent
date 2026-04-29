from typing import List, Tuple


class MemoryManager:
    """功能：管理会话记忆摘要与消息窗口裁剪。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, *, max_memory_turns: int, memory_text_limit: int, max_history_messages: int):
        """功能：配置会话记忆窗口、文本截断长度与消息历史上限。
        参数：
        - max_memory_turns：记忆中保留的最近对话轮数上限。
        - memory_text_limit：单条记忆文本截断长度上限。
        - max_history_messages：发送给模型的历史消息条数上限（不含 system）。
        返回值：
        - 无。各限制仅影响上下文裁剪，不会改写外部持久化记录。
        """
        self.max_memory_turns = max_memory_turns
        self.memory_text_limit = memory_text_limit
        self.max_history_messages = max_history_messages

    def clip_text(self, text: str) -> str:
        """功能：压缩并截断文本，防止记忆过长。
        参数：
        - text：待处理文本内容。
        返回值：
        - str：处理后的短文本。
        """
        value = (text or "").strip().replace("\n", " ")
        if len(value) <= self.memory_text_limit:
            return value
        return value[: self.memory_text_limit] + "...(truncated)"

    def build_recent_memory_context(self, turns: List[Tuple[str, str]]) -> str:
        """功能：把最近若干轮对话整理为可注入提示词的上下文文本。
        参数：
        - turns：历史问答轮次列表。
        返回值：
        - str：编号后的最近对话摘要文本；无历史时返回空串。
        """
        if not turns:
            return ""
        recent_turns = turns[-self.max_memory_turns :] if self.max_memory_turns > 0 else turns
        lines = []
        for idx, (question, answer) in enumerate(recent_turns, start=1):
            lines.append(f"{idx}. 用户：{self.clip_text(question)}")
            lines.append(f"   助手：{self.clip_text(answer)}")
        return "\n".join(lines)

    def remember_turn(self, turns: List[Tuple[str, str]], user_input: str, final_answer: str) -> None:
        """功能：记录一轮问答并按容量限制裁剪旧记录。
        参数：
        - turns：历史问答轮次列表（原地更新）。
        - user_input：当前用户输入文本。
        - final_answer：模型最终回复文本。
        返回值：
        - 无。
        """
        if not user_input:
            return
        turns.append((user_input, final_answer or ""))
        if self.max_memory_turns > 0 and len(turns) > self.max_memory_turns:
            del turns[:-self.max_memory_turns]

    def trim_messages(self, messages: List[dict]) -> None:
        """功能：裁剪消息历史，保留 system 消息与最近若干轮对话。
        参数：
        - messages：与模型交互的消息列表。
        返回值：
        - 无。
        """
        if self.max_history_messages <= 0 or not messages:
            return
        system_msg = messages[0]
        conversation = messages[1:]
        if len(conversation) <= self.max_history_messages:
            return
        kept = [system_msg] + conversation[-self.max_history_messages :]
        messages[:] = kept
