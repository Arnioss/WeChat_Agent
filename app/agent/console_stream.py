"""CLI / Web / 企微入口共用的控制台流式输出回调。"""



from __future__ import annotations



import os

import re

from typing import Callable, Tuple



from app.agent.kb_body_images import KbImageStreamSplitter, strip_kb_image_markers

from app.agent.model_client import log_answer, log_print



ConsoleStreamCallback = Callable[[str, str], None]



_KB_IMAGE_MARKER_RE = re.compile(r"\[KB_IMAGE:[^\]]+\]")





def _sanitize_console_trace(text: str) -> str:

    """功能：清理控制台 trace 中的 KB 图示标记，避免打印原始路径。
    参数：
    - text：待清理的控制台 trace 文本。
    返回值：
    - str：将 [KB_IMAGE:...] 替换为 [图示] 后的文本。
    """

    return _KB_IMAGE_MARKER_RE.sub("[图示]", text or "")





def make_console_stream_callback(*, web_mode: bool = False) -> Tuple[ConsoleStreamCallback, KbImageStreamSplitter]:

    """功能：构建控制台流式输出回调。

    参数：

    - web_mode：Web 入口为 True 时不重复打印图示路径（页面已渲染图片），并清理 KB 标记。

    返回值：

    - tuple：`(stream_callback, kb_splitter)`，供各入口在运行 Agent 时使用。

    """

    kb_splitter = KbImageStreamSplitter()

    trace_header_printed = False

    final_accumulator = ""



    def _print_image_path(path: str) -> None:

        """功能：在控制台打印知识库图示的本地路径与 file:// 链接。
        参数：
        - path：图示文件的本地路径。
        返回值：
        - 无。
        """

        abs_path = os.path.abspath(path.strip())

        file_url = "file:///" + abs_path.replace("\\", "/")

        log_print(f"\n[图示] {abs_path}\n  可点击: {file_url}\n", end="", flush=True)



    def _accumulate_final_text(payload: str) -> None:

        """功能：累积 final_answer 流式文本，Web 模式下剥离 KB 标记。
        参数：
        - payload：新增的最终回答文本片段。
        返回值：
        - 无。
        """

        nonlocal final_accumulator

        text = strip_kb_image_markers(payload) if web_mode else payload

        if text:

            final_accumulator += text



    def _feed_final_chunk(chunk: str) -> None:

        """功能：将 final_answer 分片喂入 KB 图示拆分器并分发文本/图片。
        参数：
        - chunk：流式 final_answer 文本分片。
        返回值：
        - 无。
        """

        for kind, payload in kb_splitter.feed(chunk):

            if kind == "text" and payload:

                _accumulate_final_text(payload)

            elif kind == "image" and payload and not web_mode:

                _print_image_path(payload)



    def stream_callback(section: str, chunk: str) -> None:

        """功能：Agent 流式事件回调，路由 trace 与 final_answer 到控制台输出。
        参数：
        - section：事件类型（如 agent_start、final_answer、final_answer_flush）。
        - chunk：事件文本载荷。
        返回值：
        - 无。
        """

        nonlocal trace_header_printed, final_accumulator

        if section == "final_answer_flush":

            for kind, payload in kb_splitter.flush():

                if kind == "text" and payload:

                    _accumulate_final_text(payload)

                elif kind == "image" and payload and not web_mode:

                    _print_image_path(payload)

            answer_text = strip_kb_image_markers(final_accumulator) if web_mode else final_accumulator

            log_answer(answer_text)

            final_accumulator = ""

            return

        if not chunk:

            return

        if section in ("agent_start", "model_decision", "tool_start", "tool_end", "agent_finish", "error"):

            if not trace_header_printed:

                log_print("执行过程：")

                trace_header_printed = True

            text = _sanitize_console_trace(str(chunk or "")) if web_mode else str(chunk or "")

            log_print(text)

            return

        if section == "final_answer":

            _feed_final_chunk(chunk)

            return



    return stream_callback, kb_splitter

