import ast
import json
from typing import Optional


def try_parse_wecom_payload(answer_text: str) -> Optional[dict]:
    """功能：从模型回复文本中提取企业微信消息负载字典。
    参数：
    - answer_text：待解析的原始文本，可能是 JSON、Python 字面量或夹杂其他文字。
    返回值：
    - Optional[dict]：成功解析且包含 `msgtype` 时返回消息字典，否则返回 None。
    """
    if not answer_text:
        return None
    s = str(answer_text).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("msgtype"):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and obj.get("msgtype"):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    sub = s[start : end + 1]
    try:
        obj = json.loads(sub)
        if isinstance(obj, dict) and obj.get("msgtype"):
            return obj
    except Exception:
        try:
            obj = ast.literal_eval(sub)
            if isinstance(obj, dict) and obj.get("msgtype"):
                return obj
        except Exception:
            return None
    return None


def extract_user_id(msg: dict) -> str:
    """功能：从企业微信消息结构中提取发送者用户标识。
    参数：
    - msg：企业微信上行消息字典。
    返回值：
    - str：优先返回 `from.userid`，其次 `from.external_userid` 或顶层 `userid`，都不存在时返回 `unknown_user`。
    """
    from_obj = msg.get("from") if isinstance(msg.get("from"), dict) else {}
    return (
        from_obj.get("userid")
        or from_obj.get("external_userid")
        or msg.get("userid")
        or "unknown_user"
    )


def extract_session_id(msg: dict) -> str:
    """功能：从消息中解析会话标识，用于关联上下文。
    参数：
    - msg：企业微信上行消息字典。
    返回值：
    - str：优先返回 `chatid`，其次 `conversation_id` 或 `session_id`，都不存在时返回 `default`。
    """
    return (
        msg.get("chatid")
        or msg.get("conversation_id")
        or msg.get("session_id")
        or "default"
    )
