import ast
import json
from typing import Optional


def try_parse_wecom_payload(answer_text: str) -> Optional[dict]:
    """功能：从模型输出中提取企业微信可下发载荷（兼容 JSON、字典字面量和包裹文本）。
    参数：
    - answer_text：模型生成的原始文本。
    返回值：
    - Optional[dict]：成功解析到且包含 msgtype 时返回载荷字典，否则返回 None。
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
    """功能：从企业微信消息对象中提取用户标识。
    参数：
    - msg：消息对象。
    返回值：
    - str：提取到的用户ID；无法获取时返回 "unknown_user"。
    """
    from_obj = msg.get("from") if isinstance(msg.get("from"), dict) else {}
    return (
        from_obj.get("userid")
        or from_obj.get("external_userid")
        or msg.get("userid")
        or "unknown_user"
    )


def extract_session_id(msg: dict) -> str:
    """功能：从企业微信消息对象中提取会话标识。
    参数：
    - msg：消息对象。
    返回值：
    - str：会话ID；缺失时返回默认值 "default"。
    """
    return (
        msg.get("chatid")
        or msg.get("conversation_id")
        or msg.get("session_id")
        or "default"
    )
