# NanoGPT-ZH 更新日志

@Author xiaomin.zhang

## [2026-02-09] 项目重命名为 NanoGPT-ZH

### 项目命名
- 正式命名为 **NanoGPT-ZH**
- Nano（纳米/微型）+ GPT + ZH（中文）
- 致敬 Andrej Karpathy 的 nanoGPT
- 更新所有文档中的项目名称和描述

### 品牌标识
- 添加项目徽章（Python、PyTorch、License）
- 统一项目命名规范
- 完善项目介绍和特点说明

---

## [2026-02-09] 项目结构优化

### 重大变更

#### 目录结构重组

**新增目录:**
- `scripts/` - 数据处理和辅助脚本
- `tests/` - 测试代码
- `examples/` - 示例代码
- `docs/` - 项目文档（已存在，进一步完善）

**文件移动:**

1. **数据处理脚本 → scripts/**
   - `extract_wiki.py` → `scripts/extract_wiki.py`
   - `prepare_data.py` → `scripts/prepare_data.py`
   - `process_wiki_data.py` → `scripts/process_wiki_data.py`

2. **测试代码 → tests/**
   - `test_model.py` → `tests/test_model.py`
   - `test_gpu.py` → `tests/test_gpu.py`
   - `install_pytorch_gpu.py` → `tests/install_pytorch_gpu.py`

3. **示例代码 → examples/**
   - `inference.py` → `examples/inference.py`

4. **文档 → docs/**
   - `MODEL_PARAMETERS_GUIDE.md` → `docs/MODEL_PARAMETERS_GUIDE.md`
   - `TRANSFORMER_ARCHITECTURE.md` → `docs/TRANSFORMER_ARCHITECTURE.md`

**新增文件:**
- `.gitignore` - Git版本控制忽略规则
- `docs/PROJECT_STRUCTURE.md` - 详细的项目结构说明
- `docs/CHANGELOG.md` - 更新日志（本文件）
- `PROJECT_STRUCTURE_SUMMARY.md` - 项目结构快速参考
- `data/raw/.gitkeep` - 保留空目录
- `data/processed/.gitkeep` - 保留空目录
- `checkpoints/.gitkeep` - 保留空目录

### Git管理优化

**新增 .gitignore 规则:**
- Python缓存文件（`__pycache__/`, `*.pyc`）
- 虚拟环境（`.venv/`, `venv/`）
- 模型文件（`*.pt`, `*.pth`）
- 训练日志（`logs/`, `*.log`）
- 数据文件（`data/raw/*`, `data/processed/*.txt`）
- IDE配置（`.vscode/`, `.idea/`）
- 临时文件和输出

### 文档更新

**更新 README.md:**
- 更新项目结构说明
- 更新快速开始指南
- 更新文件路径引用

**新增文档:**
- `docs/PROJECT_STRUCTURE.md` - 详细的项目结构和开发指南
- `PROJECT_STRUCTURE_SUMMARY.md` - 快速参考
- `docs/CHANGELOG.md` - 更新日志

### 优化效果

1. **更清晰的结构**
   - 按功能和类型组织代码
   - 核心代码、工具、脚本、测试、示例分离
   - 文档集中管理

2. **更好的可维护性**
   - 模块职责明确
   - 易于查找和修改
   - 便于团队协作

3. **更规范的Git管理**
   - 忽略大文件和临时文件
   - 保留必要的目录结构
   - 只提交源代码和文档

4. **更友好的开发体验**
   - 清晰的文档指引
   - 标准的项目结构
   - 完善的开发工作流

### 兼容性说明

**影响的命令:**

旧命令 → 新命令
- `python prepare_data.py` → `python scripts/prepare_data.py`
- `python extract_wiki.py` → `python scripts/extract_wiki.py`
- `python inference.py` → `python examples/inference.py`
- `python test_model.py` → `python tests/test_model.py`
- `python test_gpu.py` → `python tests/test_gpu.py`

**不受影响的命令:**
- `python train.py` - 保持不变
- `tensorboard --logdir=logs` - 保持不变

### 后续计划

- [ ] 添加单元测试框架（pytest）
- [ ] 添加CI/CD配置
- [ ] 添加Docker支持
- [ ] 完善API文档
- [ ] 添加更多示例

---

## 历史版本

### [2026-02-08] 初始版本
- 实现基础GPT模型
- 实现训练和推理流程
- 支持中文文本生成
- 添加维基百科数据处理
- GPU支持和优化

### [2026-02-07] 项目启动
- 项目初始化
- 确定技术方案
- 搭建基础框架
