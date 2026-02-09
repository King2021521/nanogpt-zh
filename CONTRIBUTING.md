# 贡献指南 | Contributing Guide

感谢您对 **NanoGPT-ZH** 项目的关注！

## 🌟 如何贡献

### 报告问题 (Issues)

如果您发现了bug或有功能建议：

1. 检查是否已有相关issue
2. 创建新issue，使用清晰的标题和描述
3. 提供复现步骤（如果是bug）
4. 附上环境信息（Python版本、PyTorch版本等）

### 提交代码 (Pull Requests)

1. **Fork 项目**
   ```bash
   git clone https://github.com/your-username/nanogpt-zh.git
   cd nanogpt-zh
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **编写代码**
   - 遵循项目代码风格
   - 添加必要的注释（携带 `@Author xiaomin.zhang`）
   - 编写测试用例

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```

5. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 代码规范

### Python 代码风格
- 遵循 PEP 8
- 使用有意义的变量名
- 函数和类添加docstring
- 所有注释携带作者信息

```python
"""
模块说明
@Author xiaomin.zhang
"""

def function_name(param1, param2):
    """
    函数说明
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    
    @Author xiaomin.zhang
    """
    pass
```

### 提交信息规范

使用语义化的提交信息：

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat: 添加多GPU训练支持
fix: 修复tokenizer的边界情况bug
docs: 更新README安装说明
```

## 🎯 贡献方向

### 欢迎的贡献类型

1. **Bug修复**
   - 修复已知问题
   - 改进错误处理

2. **功能增强**
   - 添加新的采样策略
   - 支持更多数据格式
   - 优化训练性能

3. **文档改进**
   - 完善API文档
   - 添加使用示例
   - 翻译文档

4. **测试**
   - 添加单元测试
   - 提高代码覆盖率

5. **性能优化**
   - 提升训练速度
   - 减少内存占用

### 不建议的贡献

- 引入大型外部依赖
- 破坏现有API的向后兼容性
- 未经讨论的重大架构变更

## 🧪 测试

提交PR前请确保：

```bash
# 运行测试
python tests/test_model.py

# 检查GPU支持
python tests/test_gpu.py
```

## 📚 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-username/nanogpt-zh.git
cd nanogpt-zh

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行测试
python tests/test_model.py
```

## 💬 交流讨论

- **GitHub Issues**: 报告bug和功能请求
- **GitHub Discussions**: 一般性讨论和问答
- **Pull Requests**: 代码审查和讨论

## 📄 许可证

贡献的代码将采用与项目相同的 MIT 许可证。

## 🙏 致谢

感谢所有贡献者！您的每一个贡献都让 NanoGPT-ZH 变得更好。

---

<div align="center">

**NanoGPT-ZH** - 从零手搓的中文GPT模型

Made with ❤️ by the community

</div>
