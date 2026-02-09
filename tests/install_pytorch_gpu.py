"""
安装支持GPU的PyTorch版本
@Author xiaomin.zhang
"""

import subprocess
import sys

print("=" * 60)
print("安装支持CUDA的PyTorch")
print("=" * 60)

print("\n检测到的信息:")
print("  显卡: NVIDIA GeForce GT 710")
print("  驱动版本: 456.71")
print("  CUDA版本: 11.1")
print("  显卡计算能力: 3.5")

print("\n注意事项:")
print("  - GT 710是较老的显卡（Kepler架构）")
print("  - 计算能力3.5，PyTorch 2.0+已不支持")
print("  - 建议安装PyTorch 1.13.1（最后支持compute 3.5的版本）")

print("\n" + "=" * 60)
choice = input("是否继续安装PyTorch 1.13.1 + CUDA 11.7? (y/n): ")

if choice.lower() != 'y':
    print("安装取消")
    sys.exit(0)

print("\n开始安装...")

# 1. 卸载当前PyTorch
print("\n1. 卸载当前PyTorch版本...")
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=True)
    print("✓ 卸载完成")
except:
    print("注：如果未安装，卸载会跳过")

# 2. 安装支持CUDA的PyTorch 1.13.1
print("\n2. 安装PyTorch 1.13.1 (CUDA 11.7)...")
print("   这可能需要几分钟时间...")

install_cmd = [
    sys.executable, "-m", "pip", "install",
    "torch==1.13.1+cu117",
    "torchvision==0.14.1+cu117", 
    "torchaudio==0.13.1+cu117",
    "--index-url", "https://download.pytorch.org/whl/cu117"
]

try:
    result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
    print("✓ 安装完成")
except subprocess.CalledProcessError as e:
    print(f"✗ 安装失败: {e}")
    print("请手动运行以下命令:")
    print(" ".join(install_cmd))
    sys.exit(1)

print("\n3. 验证安装...")
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU名称: {torch.cuda.get_device_name(0)}")
        print("\n🎉 安装成功！现在可以使用GPU训练了！")
    else:
        print("\n⚠ 警告: CUDA仍不可用，可能需要重启Python环境")
except Exception as e:
    print(f"✗ 验证失败: {e}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
