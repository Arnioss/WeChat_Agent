import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.vector_store import VectorStoreService


def main() -> None:
    """功能：重建或增量更新本地 RAG 向量索引并输出统计信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    stats = VectorStoreService().build_or_update_index()
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
