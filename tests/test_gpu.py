"""
GPU可用性测试脚本
@Author xiaomin.zhang
"""

import sys
import io

# 设置标准输出编码为UTF-8（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("GPU 可用性测试")
print("=" * 60)

# 1. 检查PyTorch是否安装
print("\n1. 检查PyTorch安装...")
try:
    import torch
    print(f"✓ PyTorch已安装，版本: {torch.__version__}")
except ImportError:
    print("✗ PyTorch未安装！")
    sys.exit(1)

# 2. 检查CUDA是否可用
print("\n2. 检查CUDA支持...")
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available}")

if not cuda_available:
    print("\n⚠ CUDA不可用，可能的原因：")
    print("  1. 安装的是CPU版本的PyTorch")
    print("  2. 未安装NVIDIA CUDA驱动")
    print("  3. GPU不支持CUDA")
    print("\n检查详细信息：")
else:
    print("✓ CUDA可用！")

# 3. 显示CUDA详细信息
print(f"\nPyTorch内置CUDA版本: {torch.version.cuda if torch.version.cuda else 'None (CPU版本)'}")
print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print(f"cuDNN是否启用: {torch.backends.cudnn.enabled if torch.cuda.is_available() else 'N/A'}")

# 4. 显示GPU信息
if cuda_available:
    print("\n3. GPU设备信息...")
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        print(f"\n--- GPU {i} ---")
        print(f"名称: {torch.cuda.get_device_name(i)}")
        print(f"计算能力: {torch.cuda.get_device_capability(i)}")
        
        # 显存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"总显存: {total_memory:.2f} GB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"已分配显存: {allocated:.2f} GB")
            print(f"缓存显存: {cached:.2f} GB")
else:
    print("\n3. GPU设备信息...")
    print("未检测到可用的GPU设备")

# 5. 测试GPU计算
print("\n4. 测试GPU计算...")
if cuda_available:
    try:
        # 创建测试张量
        x = torch.randn(1000, 1000)
        print(f"创建CPU张量: shape={x.shape}")
        
        # 移动到GPU
        x_gpu = x.cuda()
        print(f"✓ 成功将张量移动到GPU")
        
        # GPU计算
        import time
        start = time.time()
        y_gpu = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()  # 等待GPU计算完成
        gpu_time = time.time() - start
        print(f"✓ GPU矩阵乘法完成，耗时: {gpu_time*1000:.2f} ms")
        
        # CPU计算对比
        start = time.time()
        y_cpu = torch.matmul(x, x)
        cpu_time = time.time() - start
        print(f"✓ CPU矩阵乘法完成，耗时: {cpu_time*1000:.2f} ms")
        
        print(f"\n速度提升: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"✗ GPU计算测试失败: {e}")
else:
    print("跳过GPU计算测试（CUDA不可用）")

# 6. 检查显卡驱动和CUDA版本兼容性
print("\n5. 兼容性检查...")
if cuda_available:
    print("✓ PyTorch与GPU兼容")
else:
    print("\n您的显卡: NVIDIA GeForce GT 710")
    print("\n建议操作：")
    print("1. 检查NVIDIA驱动是否已安装：")
    print("   - 在命令行运行: nvidia-smi")
    print("   - 如果显示GPU信息，说明驱动已安装")
    print("\n2. 重新安装支持CUDA的PyTorch：")
    print("   - 卸载当前版本: pip uninstall torch torchvision")
    print("   - 安装CUDA版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   注：GT 710支持CUDA，但需要匹配的PyTorch版本")
    print("\n3. 检查显卡计算能力：")
    print("   - GT 710的计算能力约为3.5")
    print("   - 确保安装的PyTorch版本支持该计算能力")

# 7. 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
if cuda_available:
    print("✓ GPU可用，可以进行GPU训练！")
    print(f"✓ 建议在config.py中确认device设置为'cuda'")
else:
    print("✗ GPU不可用，当前只能使用CPU训练")
    print("✗ 请按照上述建议重新安装PyTorch CUDA版本")

print("\n当前设备配置:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  torch.device: {device}")
print("=" * 60)
