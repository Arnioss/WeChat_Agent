from typing import List


def build_stream_reply(content: str, stream_id: str, finish: bool = False) -> dict:
    """功能：构建企业微信流式回复消息体。
    参数：
    - content：本次流式片段要展示的文本内容。
    - stream_id：流式会话标识，用于把多次推送归并到同一条消息。
    - finish：是否为最后一个片段；True 表示流式输出结束。
    返回值：
    - dict：符合企业微信 `stream` 消息结构的字典。
    """
    return {
        "msgtype": "stream",
        "stream": {
            "id": stream_id,
            "finish": finish,
            "content": content,
        },
    }


def build_stream_with_template_card_reply(
    *,
    content: str,
    stream_id: str,
    finish: bool,
    template_card: dict,
) -> dict:
    """功能：构建同时包含流式内容与模板卡片的回复消息体。
    参数：
    - content：流式文本内容。
    - stream_id：流式会话标识。
    - finish：是否结束流式输出。
    - template_card：要附带展示的模板卡片字典。
    返回值：
    - dict：符合 `stream_with_template_card` 协议的消息字典。
    """
    return {
        "msgtype": "stream_with_template_card",
        "stream": {
            "id": stream_id,
            "finish": finish,
            "content": content,
        },
        "template_card": template_card,
    }


def build_multiple_interaction_template_card(
    *,
    task_id: str,
    main_title_title: str,
    main_title_desc: str,
    select_list: List[dict],
    submit_text: str,
    submit_key: str,
) -> dict:
    """功能：构建多选交互模板卡片消息。
    参数：
    - task_id：模板卡片任务 ID，用于后续更新同一张卡片。
    - main_title_title：卡片主标题文本。
    - main_title_desc：卡片主标题补充说明。
    - select_list：题目配置列表，每项包含问题键、标题和选项列表。
    - submit_text：提交按钮展示文本。
    - submit_key：提交按钮回传键值。
    返回值：
    - dict：可直接发送的 `template_card` 消息字典。
    """
    card = {
        "card_type": "multiple_interaction",
        "source": {
            "icon_url": "https://wework.qpic.cn/wwpic/252813_jOfDHtcISzuodLa_1629280209/0",
            "desc": "企业微信",
        },
        "main_title": {
            "title": main_title_title,
            "desc": main_title_desc,
        },
        "select_list": [],
        "submit_button": {"text": submit_text, "key": submit_key},
        "task_id": task_id,
    }

    for sel in select_list:
        question_key = sel.get("question_key")
        options = sel.get("option_list") or []
        if not question_key or not options:
            continue
        first_id = options[0].get("id")
        card["select_list"].append(
            {
                "question_key": question_key,
                "title": sel.get("title") or question_key,
                "disable": False,
                "selected_id": sel.get("selected_id") or first_id,
                "option_list": [
                    {"id": str(o.get("id")), "text": str(o.get("text") or o.get("id"))}
                    for o in options
                ],
            }
        )

    return {"msgtype": "template_card", "template_card": card}


def build_text_reply(content: str) -> dict:
    """功能：构建普通文本回复消息体。
    参数：
    - content：要发送给用户的文本内容。
    返回值：
    - dict：符合企业微信文本消息结构的字典。
    """
    return {"msgtype": "text", "text": {"content": content}}


def build_update_template_card_text_notice(
    *,
    task_id: str,
    userids: List[str],
    content: str,
) -> dict:
    """功能：构建模板卡片更新通知消息，用于刷新任务结果。
    参数：
    - task_id：要更新的模板卡片任务 ID。
    - userids：需要接收更新通知的用户 ID 列表。
    - content：展示在更新卡片中的结果文本。
    返回值：
    - dict：符合 `update_template_card` 接口要求的消息字典。
    """
    c = str(content or "").replace("\n", " ").strip()
    c = c[:112]
    if not c:
        c = "操作完成"

    card = {
        "card_type": "text_notice",
        "source": {"desc": "企业微信"},
        "main_title": {"title": "车辆借用助手", "desc": "操作结果"},
        "sub_title_text": c,
        "card_action": {"type": 1, "url": "https://work.weixin.qq.com/?from=openApi"},
        "task_id": str(task_id or ""),
    }
    return {
        "response_type": "update_template_card",
        "userids": userids,
        "template_card": card,
    }
