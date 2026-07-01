from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """功能：禁用 Chroma 产品遥测，避免引入 posthog 依赖。
    参数：
    - 无。
    返回值：
    - 无。
    """

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        """功能：忽略遥测事件，不向外部上报。
        参数：
        - event：Chroma 遥测事件对象。
        返回值：
        - 无。
        """
        return None
