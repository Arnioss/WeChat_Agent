import json
import logging
import threading
import time
from typing import Any, Dict, Optional


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """功能：按统一 JSON 结构记录业务事件日志。
    参数：
    - logger：目标日志记录器实例。
    - event：事件名称。
    - fields：事件附加字段。
    返回值：
    - 无。
    """
    payload = {"event": event, **fields}
    logger.info("事件 %s", json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))


def observe_phase(metrics: "AppMetrics", phase: str, duration_ms: float) -> None:
    """功能：记录单阶段耗时到指标系统。
    参数：
    - metrics：指标聚合器实例。
    - phase：阶段名称（如 session_get、arun）。
    - duration_ms：耗时毫秒数。
    返回值：
    - 无。
    """
    if duration_ms < 0:
        return
    safe = AppMetrics._sanitize(phase)
    metrics.observe_ms(f"ws_phase_{safe}", duration_ms)


def log_timing(
    logger: logging.Logger,
    metrics: "AppMetrics",
    event: str,
    *,
    phases: Optional[Dict[str, float]] = None,
    total_ms: Optional[float] = None,
    **fields: Any,
) -> None:
    """功能：同时写入结构化耗时事件日志与各阶段指标。
    参数：
    - logger：目标日志记录器。
    - metrics：指标聚合器实例。
    - event：事件名称。
    - phases：各阶段耗时毫秒字典。
    - total_ms：总耗时毫秒数。
    - fields：附加字段。
    返回值：
    - 无。
    """
    if phases:
        for name, value in phases.items():
            observe_phase(metrics, name, float(value))
    if total_ms is not None and total_ms >= 0:
        observe_phase(metrics, "total", float(total_ms))
    payload: Dict[str, Any] = dict(fields)
    if phases:
        payload["phases_ms"] = {k: round(float(v), 2) for k, v in phases.items()}
    if total_ms is not None:
        payload["total_ms"] = round(float(total_ms), 2)
    log_event(logger, event, **payload)


class AppMetrics:
    """功能：提供线程安全的计数器与仪表盘指标聚合能力。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self):
        """功能：初始化线程安全指标容器，维护计数器与仪表盘两类指标。
        参数：
        - 无。
        返回值：
        - 无。内部使用互斥锁保证并发更新时的读写一致性。
        """
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        """功能：按名称递增计数器。
        参数：
        - name：计数器名称。
        - value：递增值，默认 1。
        返回值：
        - 无。
        """
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """功能：设置仪表盘数值指标。
        参数：
        - name：指标名称。
        - value：最新指标值。
        返回值：
        - 无。
        """
        with self._lock:
            self._gauges[name] = value

    def observe_ms(self, name: str, value_ms: float) -> None:
        """功能：记录毫秒级耗时观测值并累计统计。
        参数：
        - name：指标前缀名称。
        - value_ms：本次耗时（毫秒）。
        返回值：
        - 无。
        """
        with self._lock:
            self._counters[f"{name}_sum_ms"] = self._counters.get(f"{name}_sum_ms", 0.0) + value_ms
            self._counters[f"{name}_count"] = self._counters.get(f"{name}_count", 0.0) + 1.0

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """功能：获取当前指标快照。
        参数：
        - 无。
        返回值：
        - Dict[str, Dict[str, float]]：包含 counters、gauges 与时间戳的快照字典。
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timestamp": {"unix": time.time()},
            }

    def render_prometheus(self) -> str:
        """功能：将当前指标渲染为 Prometheus 文本格式。
        参数：
        - 无。
        返回值：
        - str：Prometheus exposition 格式文本。
        """
        lines = []
        snap = self.snapshot()
        for name, value in sorted(snap["counters"].items()):
            metric_name = self._sanitize(name)
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")
        for name, value in sorted(snap["gauges"].items()):
            metric_name = self._sanitize(name)
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        lines.append(f"wecom_bot_metrics_timestamp_unix {snap['timestamp']['unix']}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _sanitize(name: str) -> str:
        """功能：把指标名转换为 Prometheus 兼容格式。
        参数：
        - name：原始指标名。
        返回值：
        - str：仅包含小写字母、数字和下划线的安全指标名。
        """
        safe = []
        for ch in name:
            if ch.isalnum() or ch == "_":
                safe.append(ch.lower())
            else:
                safe.append("_")
        return "".join(safe)


def bind_llm_metrics(metrics: AppMetrics) -> None:
    """功能：将 llm_key_pool 指标钩子绑定到 AppMetrics。
    参数：
    - metrics：应用指标聚合器实例。
    返回值：
    - 无。
    """
    from app.agent.llm_key_pool import set_llm_metrics_hook

    def hook(event: str, value: float) -> None:
        """功能：将 llm_key_pool 事件转发到 AppMetrics 计数器、仪表盘或耗时指标。
        参数：
        - event：事件名，前缀 inc:/gauge:/observe_ms: 决定处理方式。
        - value：指标数值。
        返回值：
        - 无。
        """
        if event.startswith("inc:"):
            metrics.inc(event[4:], value)
        elif event.startswith("gauge:"):
            metrics.set_gauge(event[6:], value)
        elif event.startswith("observe_ms:"):
            metrics.observe_ms(event[11:], value)

    set_llm_metrics_hook(hook)
