import json
import logging
from typing import Any, Callable

from flask import Response

from wxwork.crypto import WXBizJsonMsgCrypt


class WeComCryptoService:
    """功能：封装企业微信回包加密与响应构造流程。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        token: str,
        encoding_aes_key: str,
        receive_id: str,
        reply_debug_log: bool,
        build_stream_reply: Callable[..., dict],
        safe_json: Callable[[Any], str],
        logger: logging.Logger,
    ):
        """功能：注入企业微信加密配置与回包构造依赖，建立统一加密服务实例。
        参数：
        - token：企业微信回调 token。
        - encoding_aes_key：企业微信消息加密密钥。
        - receive_id：企业微信接收方 ID（CorpID 或 SuiteID）。
        - reply_debug_log：是否记录回包调试日志。
        - build_stream_reply：构建流式回包消息体的函数。
        - safe_json：安全序列化 JSON 的函数。
        - logger：日志记录器实例。
        返回值：
        - 无。该实例本身不持有会话态，每次请求通过 `new_crypt` 创建新的加解密对象。
        """
        self.token = token
        self.encoding_aes_key = encoding_aes_key
        self.receive_id = receive_id
        self.reply_debug_log = reply_debug_log
        self.build_stream_reply = build_stream_reply
        self.safe_json = safe_json
        self.logger = logger

    def new_crypt(self) -> WXBizJsonMsgCrypt:
        """功能：创建企业微信 JSON 消息加解密对象。
        参数：
        - 无。
        返回值：
        - WXBizJsonMsgCrypt：可用于验签和加密回包的对象。
        异常：
        - ValueError：关键加密配置缺失时抛出。
        """
        if not self.token or not self.encoding_aes_key:
            raise ValueError("未配置 WECHAT_ROBOT_TOKEN 或 WECHAT_ROBOT_ENCODING_AES_KEY")
        return WXBizJsonMsgCrypt(
            self.token,
            self.encoding_aes_key,
            self.receive_id,
        )

    def encrypt_reply(
        self,
        crypt: WXBizJsonMsgCrypt,
        plaintext_json: dict,
        nonce: str,
        timestamp: str,
    ) -> Response:
        """功能：将明文回包加密并返回 Flask 响应对象。
        参数：
        - crypt：企业微信加解密对象。
        - plaintext_json：待加密的明文消息字典。
        - nonce：请求随机串。
        - timestamp：请求时间戳。
        返回值：
        - Response：加密成功返回 JSON 响应，失败时返回 `success` 文本响应。
        """
        if self.reply_debug_log:
            self.logger.info("WECHAT_PLAINTEXT_REPLY:\n%s", self.safe_json(plaintext_json))

        ret, encrypted = crypt.EncryptMsg(
            json.dumps(plaintext_json, ensure_ascii=False),
            nonce,
            timestamp,
        )

        if self.reply_debug_log:
            self.logger.info("WECHAT_ENCRYPT_REPLY ret=%s", ret)
            if encrypted:
                self.logger.info("WECHAT_ENCRYPTED_REPLY:\n%s", encrypted)

        if ret != 0 or not encrypted:
            self.logger.warning("加密回包失败 ret=%s", ret)
            return Response("success", mimetype="text/plain")

        return Response(encrypted, mimetype="application/json")

    def encrypt_reply_text_as_stream(
        self,
        crypt: WXBizJsonMsgCrypt,
        content: str,
        nonce: str,
        timestamp: str,
        stream_id: str,
    ) -> Response:
        """功能：把纯文本包装为 stream 消息并加密返回。
        参数：
        - crypt：企业微信加解密对象。
        - content：要发送的文本内容。
        - nonce：请求随机串。
        - timestamp：请求时间戳。
        - stream_id：流式消息 ID。
        返回值：
        - Response：加密后的 Flask 响应对象。
        """
        reply = self.build_stream_reply(content=str(content or ""), stream_id=stream_id, finish=True)
        return self.encrypt_reply(crypt, reply, nonce, timestamp)
