#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Source reference:
# - WeCom (Enterprise WeChat) Developer Documentation
# - "回调和回复的加解密方案"
# - https://developer.work.weixin.qq.com/document/path/101033
#
# Notes:
# - This file is an implementation used for WeCom callback decrypt/encrypt flows.
# - Adapted for this project structure and error-handling conventions.

import base64
import hashlib
import json
import random
import socket
import struct
import time

from Crypto.Cipher import AES

from . import ierror


class FormatException(Exception):
    """功能：表示企业微信加解密流程中的格式错误（如密钥不合法、报文结构异常）。
    参数：
    - 无。
    返回值：
    - 无。该类仅用于异常语义标识，抛出后由上层统一返回企业微信错误码。
    """
    pass


def throw_exception(message, exception_class=FormatException):
    """功能：按指定异常类型抛出异常。
    参数：
    - message：异常消息文本。
    - exception_class：要抛出的异常类型，默认 `FormatException`。
    返回值：
    - 无。
    """
    raise exception_class(message)


class SHA1:
    """功能：提供企业微信协议要求的 SHA1 签名计算能力。
    参数：
    - 无。
    返回值：
    - 无。计算失败时由方法返回标准错误码而非直接抛异常。
    """
    def getSHA1(self, token, timestamp, nonce, encrypt):
        """功能：生成企业微信消息签名 SHA1 值。
        参数：
        - token：企业微信回调配置的 token。
        - timestamp：请求时间戳。
        - nonce：请求随机串。
        - encrypt：待签名的加密文本。
        返回值：
        - tuple[int, Optional[str]]：成功返回 `(0, signature)`，失败返回错误码和 None。
        """
        try:
            if isinstance(encrypt, bytes):
                encrypt = encrypt.decode("utf-8")
            sortlist = [str(token), str(timestamp), str(nonce), str(encrypt)]
            sortlist.sort()
            sha = hashlib.sha1()
            sha.update("".join(sortlist).encode("utf-8"))
            return ierror.WXBizMsgCrypt_OK, sha.hexdigest()
        except Exception:
            return ierror.WXBizMsgCrypt_ComputeSignature_Error, None


class JsonParse:
    """功能：处理企业微信加密报文的 JSON 提取与回包序列化。
    参数：
    - 无。
    返回值：
    - 无。仅处理协议字段，不负责签名校验与加解密。
    """
    AES_TEXT_RESPONSE_TEMPLATE = """{
        "encrypt": "%(msg_encrypt)s",
        "msgsignature": "%(msg_signaturet)s",
        "timestamp": "%(timestamp)s",
        "nonce": "%(nonce)s"
    }"""

    def extract(self, jsontext):
        """功能：从回调 JSON 文本中提取 `encrypt` 字段。
        参数：
        - jsontext：企业微信回调消息 JSON 字符串。
        返回值：
        - tuple[int, Optional[str]]：成功返回 `(0, encrypt)`，解析失败返回错误码和 None。
        """
        try:
            json_dict = json.loads(jsontext)
            return ierror.WXBizMsgCrypt_OK, json_dict["encrypt"]
        except Exception:
            return ierror.WXBizMsgCrypt_ParseJson_Error, None

    def generate(self, encrypt, signature, timestamp, nonce):
        """功能：生成企业微信加密响应 JSON 文本。
        参数：
        - encrypt：加密后的消息体文本。
        - signature：消息签名。
        - timestamp：时间戳。
        - nonce：随机串。
        返回值：
        - str：符合企业微信协议的 JSON 字符串。
        """
        resp_dict = {
            "msg_encrypt": encrypt,
            "msg_signaturet": signature,
            "timestamp": timestamp,
            "nonce": nonce,
        }
        return self.AES_TEXT_RESPONSE_TEMPLATE % resp_dict


class PKCS7Encoder:
    """功能：按企业微信约定的 32 字节分组实现 PKCS7 填充。
    参数：
    - 无。
    返回值：
    - 无。输入为字符串时会先转 UTF-8 字节再补齐。
    """
    block_size = 32

    def encode(self, text):
        """功能：按 PKCS7 规则为明文补齐到 32 字节分组长度。
        参数：
        - text：待处理文本内容。
        返回值：
        - bytes：完成 PKCS7 填充后的字节串。
        """
        text_length = len(text)
        amount_to_pad = self.block_size - (text_length % self.block_size)
        if amount_to_pad == 0:
            amount_to_pad = self.block_size
        pad = bytes([amount_to_pad])
        if isinstance(text, str):
            text = text.encode("utf-8")
        return text + pad * amount_to_pad


class Prpcrypt:
    """功能：执行企业微信消息体的 AES-CBC 加解密与接收方标识校验。
    参数：
    - 无。
    返回值：
    - 无。解密失败或接收方不匹配时通过错误码告知调用方。
    """
    def __init__(self, key):
        """功能：保存 AES 密钥并固定 CBC 模式，供后续加解密复用。
        参数：
        - key：AES 密钥字节串。
        返回值：
        - 无。要求 `key` 长度满足企业微信 32 字节约束（由上游初始化阶段保证）。
        """
        self.key = key
        self.mode = AES.MODE_CBC

    def encrypt(self, text, receiveid):
        """功能：按企业微信协议拼装明文并执行 AES-CBC 加密。
        参数：
        - text：待处理文本内容。
        - receiveid：接收方标识（CorpID 或 SuiteID）。
        返回值：
        - tuple[int, Optional[bytes]]：成功返回加密后的 base64 字节串，失败返回错误码和 None。
        """
        text = text.encode()
        text = (
            self.get_random_str()
            + struct.pack("I", socket.htonl(len(text)))
            + text
            + receiveid.encode()
        )
        pkcs7 = PKCS7Encoder()
        text = pkcs7.encode(text)
        cryptor = AES.new(self.key, self.mode, self.key[:16])
        try:
            ciphertext = cryptor.encrypt(text)
            return ierror.WXBizMsgCrypt_OK, base64.b64encode(ciphertext)
        except Exception:
            return ierror.WXBizMsgCrypt_EncryptAES_Error, None

    def decrypt(self, text, receiveid):
        """功能：执行企业微信消息解密并校验接收方标识。
        参数：
        - text：待处理文本内容。
        - receiveid：期望的接收方标识，用于校验消息归属。
        返回值：
        - tuple[int, Optional[str]]：成功返回解密后的明文 JSON 文本，失败返回错误码和 None。
        """
        try:
            cryptor = AES.new(self.key, self.mode, self.key[:16])
            plain_text = cryptor.decrypt(base64.b64decode(text))
        except Exception:
            return ierror.WXBizMsgCrypt_DecryptAES_Error, None
        try:
            pad = plain_text[-1]
            content = plain_text[16:-pad]
            json_len = socket.ntohl(struct.unpack("I", content[:4])[0])
            json_content = content[4 : json_len + 4].decode("utf-8")
            from_receiveid = content[json_len + 4 :].decode("utf-8")
        except Exception:
            return ierror.WXBizMsgCrypt_IllegalBuffer, None
        if from_receiveid != receiveid:
            return ierror.WXBizMsgCrypt_ValidateCorpid_Error, None
        return ierror.WXBizMsgCrypt_OK, json_content

    def get_random_str(self):
        """功能：生成企业微信加密消息头所需的 16 位随机数字串。
        参数：
        - 无。
        返回值：
        - bytes：随机数字编码后的字节串。
        """
        return str(random.randint(1000000000000000, 9999999999999999)).encode()


class WXBizJsonMsgCrypt:
    """功能：实现企业微信回调验签、消息解密与回包加密的完整协议编排。
    参数：
    - 无。
    返回值：
    - 无。对外统一返回 `(错误码, 数据)` 二元组，便于上层按协议处理。
    """
    def __init__(self, sToken, sEncodingAESKey, sReceiveId):
        """功能：解析并校验回调配置，建立后续验签与加解密所需上下文。
        参数：
        - sToken：企业微信回调 token。
        - sEncodingAESKey：企业微信提供的 EncodingAESKey。
        - sReceiveId：消息接收方 ID（CorpID 或 SuiteID）。
        返回值：
        - 无。若 `sEncodingAESKey` 不能解码为 32 字节密钥会抛出 `FormatException`。
        """
        try:
            self.key = base64.b64decode(sEncodingAESKey + "=")
            assert len(self.key) == 32
        except Exception:
            throw_exception("[error]: EncodingAESKey unvalid !", FormatException)
        self.m_sToken = sToken
        self.m_sReceiveId = sReceiveId

    def VerifyURL(self, sMsgSignature, sTimeStamp, sNonce, sEchoStr):
        """功能：校验 URL 验证请求签名并解密回显字符串。
        参数：
        - sMsgSignature：请求中的签名。
        - sTimeStamp：请求时间戳。
        - sNonce：请求随机串。
        - sEchoStr：请求中的加密回显字符串。
        返回值：
        - tuple[int, Optional[str]]：成功返回解密后的 echo 文本，失败返回错误码和 None。
        """
        sha1 = SHA1()
        ret, signature = sha1.getSHA1(self.m_sToken, sTimeStamp, sNonce, sEchoStr)
        if ret != 0:
            return ret, None
        if signature != sMsgSignature:
            return ierror.WXBizMsgCrypt_ValidateSignature_Error, None
        pc = Prpcrypt(self.key)
        ret, sReplyEchoStr = pc.decrypt(sEchoStr, self.m_sReceiveId)
        return ret, sReplyEchoStr

    def EncryptMsg(self, sReplyMsg, sNonce, timestamp=None):
        """功能：加密回复消息并生成可回传的 JSON 文本。
        参数：
        - sReplyMsg：明文回复消息。
        - sNonce：随机串。
        - timestamp：可选时间戳；为空时使用当前时间。
        返回值：
        - tuple[int, Optional[str]]：成功返回加密响应 JSON，失败返回错误码和 None。
        """
        pc = Prpcrypt(self.key)
        ret, encrypt = pc.encrypt(sReplyMsg, self.m_sReceiveId)
        if ret != 0:
            return ret, None
        encrypt = encrypt.decode("utf-8")
        if timestamp is None:
            timestamp = str(int(time.time()))
        sha1 = SHA1()
        ret, signature = sha1.getSHA1(self.m_sToken, timestamp, sNonce, encrypt)
        if ret != 0:
            return ret, None
        jsonParse = JsonParse()
        return ret, jsonParse.generate(encrypt, signature, timestamp, sNonce)

    def DecryptMsg(self, sPostData, sMsgSignature, sTimeStamp, sNonce):
        """功能：校验回调签名并解密业务消息。
        参数：
        - sPostData：回调请求体 JSON 字符串。
        - sMsgSignature：请求签名。
        - sTimeStamp：请求时间戳。
        - sNonce：请求随机串。
        返回值：
        - tuple[int, Optional[str]]：成功返回解密后的明文消息，失败返回错误码和 None。
        """
        jsonParse = JsonParse()
        ret, encrypt = jsonParse.extract(sPostData)
        if ret != 0:
            return ret, None
        sha1 = SHA1()
        ret, signature = sha1.getSHA1(self.m_sToken, sTimeStamp, sNonce, encrypt)
        if ret != 0:
            return ret, None
        if signature != sMsgSignature:
            return ierror.WXBizMsgCrypt_ValidateSignature_Error, None
        pc = Prpcrypt(self.key)
        return pc.decrypt(encrypt, self.m_sReceiveId)
