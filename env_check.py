"""深度学习环境自检脚本 —— 在 torch 环境或 tf 环境中运行都可以。

用法：
    D:\\conda-envs\\torch\\python.exe env_check.py
    D:\\conda-envs\\tf\\python.exe    env_check.py

会打印 Python / 框架 / CUDA / GPU 信息，并跑一个小型卷积网络验证 GPU 真的能算。
"""
import sys
import time
import platform


def sep(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_torch():
    try:
        import torch
    except ImportError:
        return False

    sep("PyTorch")
    print(f"版本        : {torch.__version__}")
    print(f"CUDA 编译版 : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    print(f"GPU 可用    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA 不可用")
        return True

    p = torch.cuda.get_device_properties(0)
    print(f"显卡        : {p.name}")
    print(f"显存        : {p.total_memory / 1024**3:.1f} GB")
    print(f"算力        : sm_{p.major}{p.minor}  ({p.multi_processor_count} SM)")

    dev = torch.device("cuda")
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 4),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x = torch.randn(16, 3, 512, 512, device=dev)
    y = torch.randint(0, 4, (16,), device=dev)

    # 预热
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    print(f"训练一步    : {dt * 1000:.1f} ms  (512x512, batch=16, loss={loss.item():.4f})")
    print(f"显存占用    : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("状态        : OK")
    return True


def check_tf():
    try:
        import tensorflow as tf
    except Exception as e:  # 导入失败也要报出来
        print(f"TensorFlow 导入失败: {e}")
        return False

    sep("TensorFlow")
    print(f"版本        : {tf.__version__}")
    print(f"CUDA 编译版 : {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU 数量    : {len(gpus)}")
    if not gpus:
        print("!! 没有检测到 GPU")
        return True
    for g in gpus:
        print(f"  - {g.name}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"显卡        : {details.get('device_name', 'N/A')}")
        print(f"算力        : {details.get('compute_capability', 'N/A')}")
    except Exception:
        pass

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(512, 512, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    import numpy as np
    x = np.random.rand(8, 512, 512, 3).astype("float32")
    y = np.random.randint(0, 4, (8,)).astype("int32")

    model.train_on_batch(x, y)  # 预热
    t0 = time.time()
    for _ in range(5):
        loss = model.train_on_batch(x, y)
    dt = (time.time() - t0) / 5
    print(f"训练一步    : {dt * 1000:.1f} ms  (512x512, batch=8, loss={float(loss):.4f})")
    print("状态        : OK")
    return True


if __name__ == "__main__":
    sep("运行环境")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"解释器      : {sys.executable}")
    print(f"平台        : {platform.platform()}")

    found = False
    if check_torch():
        found = True
    if check_tf():
        found = True
    if not found:
        print("\n当前环境既没有 PyTorch 也没有 TensorFlow。")
    print()

# touched by autosync test
