import json
import logging
import threading
import time
from typing import Dict


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
    logger.info("EVENT %s", json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))


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
