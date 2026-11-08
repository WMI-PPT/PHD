"""娣卞害瀛︿範鐜鑷鑴氭湰 鈥斺€?鍦?torch 鐜鎴?tf 鐜涓繍琛岄兘鍙互銆?
鐢ㄦ硶锛?    D:\\conda-envs\\torch\\python.exe env_check.py
    D:\\conda-envs\\tf\\python.exe    env_check.py

浼氭墦鍗?Python / 妗嗘灦 / CUDA / GPU 淇℃伅锛屽苟璺戜竴涓皬鍨嬪嵎绉綉缁滈獙璇?GPU 鐪熺殑鑳界畻銆?"""
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
    print(f"鐗堟湰        : {torch.__version__}")
    print(f"CUDA 缂栬瘧鐗?: {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    print(f"GPU 鍙敤    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA 涓嶅彲鐢?)
        return True

    p = torch.cuda.get_device_properties(0)
    print(f"鏄惧崱        : {p.name}")
    print(f"鏄惧瓨        : {p.total_memory / 1024**3:.1f} GB")
    print(f"绠楀姏        : sm_{p.major}{p.minor}  ({p.multi_processor_count} SM)")

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

    # 棰勭儹
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
    print(f"璁粌涓€姝?   : {dt * 1000:.1f} ms  (512x512, batch=16, loss={loss.item():.4f})")
    print(f"鏄惧瓨鍗犵敤    : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("鐘舵€?       : OK")
    return True


def check_tf():
    try:
        import tensorflow as tf
    except Exception as e:  # 瀵煎叆澶辫触涔熻鎶ュ嚭鏉?        print(f"TensorFlow 瀵煎叆澶辫触: {e}")
        return False

    sep("TensorFlow")
    print(f"鐗堟湰        : {tf.__version__}")
    print(f"CUDA 缂栬瘧鐗?: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU 鏁伴噺    : {len(gpus)}")
    if not gpus:
        print("!! 娌℃湁妫€娴嬪埌 GPU")
        return True
    for g in gpus:
        print(f"  - {g.name}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"鏄惧崱        : {details.get('device_name', 'N/A')}")
        print(f"绠楀姏        : {details.get('compute_capability', 'N/A')}")
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

    model.train_on_batch(x, y)  # 棰勭儹
    t0 = time.time()
    for _ in range(5):
        loss = model.train_on_batch(x, y)
    dt = (time.time() - t0) / 5
    print(f"璁粌涓€姝?   : {dt * 1000:.1f} ms  (512x512, batch=8, loss={float(loss):.4f})")
    print("鐘舵€?       : OK")
    return True


if __name__ == "__main__":
    sep("杩愯鐜")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"瑙ｉ噴鍣?     : {sys.executable}")
    print(f"骞冲彴        : {platform.platform()}")

    found = False
    if check_torch():
        found = True
    if check_tf():
        found = True
    if not found:
        print("\n褰撳墠鐜鏃㈡病鏈?PyTorch 涔熸病鏈?TensorFlow銆?)
    print()


