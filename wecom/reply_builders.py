from typing import List


def build_stream_reply(content: str, stream_id: str, finish: bool = False) -> dict:
    """功能：构建企业微信流式文本回复载荷。
    参数：
    - content：消息正文。
    - stream_id：流式消息ID。
    - finish：是否结束流式输出。
    返回值：
    - dict：`msgtype=stream` 的回复字典。
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
    """功能：构建“流式文本 + 模板卡片”组合回复载荷。
    参数：
    - content：消息正文。
    - stream_id：流式消息ID。
    - finish：是否结束流式输出。
    - template_card：模板卡片字典。
    返回值：
    - dict：`msgtype=stream_with_template_card` 的回复字典。
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
    """功能：构建企业微信多项交互模板卡片（multiple_interaction）。
    参数：
    - task_id：任务ID。
    - main_title_title：卡片主标题。
    - main_title_desc：卡片副标题说明。
    - select_list：卡片下拉配置列表。
    - submit_text：提交按钮文案。
    - submit_key：提交动作键。
    返回值：
    - dict：`msgtype=template_card` 的卡片回复字典。
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
    """功能：构建企业微信普通文本回复载荷。
    参数：
    - content：消息正文。
    返回值：
    - dict：`msgtype=text` 的回复字典。
    """
    return {"msgtype": "text", "text": {"content": content}}


def build_update_template_card_text_notice(
    *,
    task_id: str,
    userids: List[str],
    content: str,
) -> dict:
    """功能：构建模板卡片更新通知（text_notice），用于交互后更新指定任务卡片。
    参数：
    - task_id：任务ID。
    - userids：需要接收更新通知的用户ID列表。
    - content：消息正文。
    返回值：
    - dict：`response_type=update_template_card` 的更新载荷。
    """
    c = str(content or "").replace("\n", " ").strip()
    c = c[:112]
    if not c:
        c = "操作完成"

    card = {
        "card_type": "text_notice",
        "source": {"desc": "企业微信"},
        "main_title": {"title": "智能助手", "desc": "操作结果"},
        "sub_title_text": c,
        "card_action": {"type": 1, "url": "https://work.weixin.qq.com/?from=openApi"},
        "task_id": str(task_id or ""),
    }
    return {
        "response_type": "update_template_card",
        "userids": userids,
        "template_card": card,
    }
