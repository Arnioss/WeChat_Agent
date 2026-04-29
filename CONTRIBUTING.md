# 贡献指南

感谢你关注并参与 `WeChat_Agent`。

## 开发环境准备

1. Fork 并克隆仓库
2. 创建 Python 虚拟环境
3. 安装依赖

```bash
python -m pip install -r requirements.txt
```

4. 本地配置

- 复制 `.env.example` 为 `.env`
- 按你的本地环境填写必需变量

## 开发规范

- 保持改动聚焦、尽量小步提交
- 优先保证函数边界清晰、错误处理明确
- 严禁提交敏感信息（如 `.env`、API Key、Token、密码）
- 行为或配置变更时同步更新文档（如 `README.md`、`.env.example`）

## Pull Request 自检清单

- [ ] 代码可在本地运行
- [ ] 提交内容不包含密钥或敏感信息
- [ ] 若涉及配置/行为变更，已更新文档
- [ ] 已考虑兼容性影响
- [ ] PR 描述清楚改动范围与动机

## 提交信息建议

- 建议使用简洁的意图型提交信息
- 推荐前缀：`feat:`、`fix:`、`refactor:`、`docs:`、`chore:`、`test:`

## 安全相关说明

如果你的改动涉及安全漏洞，请遵循 `SECURITY.md` 的私下披露流程，不要在公开 Issue 中直接给出漏洞利用细节。
