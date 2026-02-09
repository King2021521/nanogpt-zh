# GPU支持安装指南

## 问题诊断

✅ **NVIDIA驱动**: 已安装（456.71）  
✅ **显卡**: GeForce GT 710（2GB显存）  
✅ **CUDA版本**: 11.1  
❌ **PyTorch**: 当前是CPU版本 (2.10.0+cpu)

## 解决方案

### 方案一：安装PyTorch 1.13.1（推荐）

由于GT 710的计算能力是3.5（Kepler架构），而PyTorch 2.0+已不再支持compute capability < 3.7的GPU，**建议安装PyTorch 1.13.1**。

#### 自动安装（推荐）

```bash
python install_pytorch_gpu.py
```

#### 手动安装

1. **卸载当前版本**
```bash
pip uninstall -y torch torchvision torchaudio
```

2. **安装PyTorch 1.13.1 + CUDA 11.7**
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

3. **验证安装**
```bash
python test_gpu.py
```

### 方案二：使用CPU训练（当前方案）

如果GPU安装遇到问题，可以继续使用CPU训练：
- ✅ 优点：无需额外配置
- ❌ 缺点：训练速度较慢（约10-50倍慢）

## 安装后验证

运行测试脚本：
```bash
python test_gpu.py
```

期望看到：
```
✓ CUDA可用: True
✓ GPU名称: GeForce GT 710
```

## 关于GT 710的性能预期

### 硬件限制
- **显存**: 2GB（较小）
- **计算能力**: 3.5（较老）
- **性能**: 入门级显卡

### 训练建议
如果成功启用GPU，建议调整配置以适应显存限制：

```python
# config.py
batch_size = 8          # 从32减小到8
d_model = 128           # 从256减小到128
n_layers = 4            # 从6减小到4
```

### 性能预期
- **GPU vs CPU**: GPU可能快2-5倍（GT 710较老）
- **训练时间**: 50K步约需2-6小时
- **显存使用**: 约1-1.5GB

## 常见问题

### Q1: 安装后CUDA仍不可用？
**A**: 重启Python环境或IDE，然后重新测试

### Q2: 训练时显存不足？
**A**: 减小batch_size或模型维度

### Q3: GPU速度提升不明显？
**A**: GT 710是入门级显卡，性能有限，正常现象

### Q4: 是否值得使用GT 710？
**A**: 
- 对于学习和理解：✅ 值得
- 对于大规模训练：❌ 建议云GPU

## 云GPU替代方案

如果本地GPU性能不足，可以考虑：

1. **Google Colab**（免费）
   - 提供Tesla T4 GPU
   - 约15GB显存
   - 网址：https://colab.research.google.com

2. **Kaggle**（免费）
   - 提供Tesla P100 GPU
   - 约16GB显存
   - 每周30小时免费额度

3. **AutoDL**（付费，国内）
   - RTX 3080: 约2元/小时
   - RTX 4090: 约3元/小时

## 下一步

1. 运行 `python install_pytorch_gpu.py` 安装GPU版本
2. 运行 `python test_gpu.py` 验证安装
3. 如果成功，修改训练配置适应显存限制
4. 重新开始训练

---

**@Author xiaomin.zhang**
