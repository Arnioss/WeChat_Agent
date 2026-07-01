import json
import sys


def main() -> None:
    """功能：输出脚本入参，便于调试技能脚本调用链路。
    参数：
    - 无。
    返回值：
    - 无。
    """
    print(json.dumps({"args": sys.argv[1:]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
