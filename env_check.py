"""濞ｅ崬瀹崇€涳缚绡勯悳顖氼暔閼奉亝顥呴懘姘拱 閳ユ柡鈧?閸?torch 閻滎垰顣ㄩ幋?tf 閻滎垰顣ㄦ稉顓＄箥鐞涘矂鍏橀崣顖欎簰閵?
閻劍纭堕敍?    D:\\conda-envs\\torch\\python.exe env_check.py
    D:\\conda-envs\\tf\\python.exe    env_check.py

娴兼碍澧﹂崡?Python / 濡楀棙鐏?/ CUDA / GPU 娣団剝浼呴敍灞借嫙鐠烘垳绔存稉顏勭毈閸ㄥ宓庣粔顖滅秹缂佹粓鐛欑拠?GPU 閻喓娈戦懗鐣岀暬閵?"""
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
    print(f"閻楀牊婀?       : {torch.__version__}")
    print(f"CUDA 缂傛牞鐦ч悧?: {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    print(f"GPU 閸欘垳鏁?   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA 娑撳秴褰查悽?)
        return True

    p = torch.cuda.get_device_properties(0)
    print(f"閺勬儳宕?       : {p.name}")
    print(f"閺勬儳鐡?       : {p.total_memory / 1024**3:.1f} GB")
    print(f"缁犳濮?       : sm_{p.major}{p.minor}  ({p.multi_processor_count} SM)")

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

    # 妫板嫮鍎?    for _ in range(3):
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
    print(f"鐠侇厾绮屾稉鈧?   : {dt * 1000:.1f} ms  (512x512, batch=16, loss={loss.item():.4f})")
    print(f"閺勬儳鐡ㄩ崡鐘垫暏    : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("閻樿埖鈧?       : OK")
    return True


def check_tf():
    try:
        import tensorflow as tf
    except Exception as e:  # 鐎电厧鍙嗘径杈Е娑旂喕顩﹂幎銉ュ毉閺?        print(f"TensorFlow 鐎电厧鍙嗘径杈Е: {e}")
        return False

    sep("TensorFlow")
    print(f"閻楀牊婀?       : {tf.__version__}")
    print(f"CUDA 缂傛牞鐦ч悧?: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU 閺佷即鍣?   : {len(gpus)}")
    if not gpus:
        print("!! 濞屸剝婀佸Λ鈧ù瀣煂 GPU")
        return True
    for g in gpus:
        print(f"  - {g.name}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"閺勬儳宕?       : {details.get('device_name', 'N/A')}")
        print(f"缁犳濮?       : {details.get('compute_capability', 'N/A')}")
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

    model.train_on_batch(x, y)  # 妫板嫮鍎?    t0 = time.time()
    for _ in range(5):
        loss = model.train_on_batch(x, y)
    dt = (time.time() - t0) / 5
    print(f"鐠侇厾绮屾稉鈧?   : {dt * 1000:.1f} ms  (512x512, batch=8, loss={float(loss):.4f})")
    print("閻樿埖鈧?       : OK")
    return True


if __name__ == "__main__":
    sep("鏉╂劘顢戦悳顖氼暔")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"鐟欙綁鍣撮崳?     : {sys.executable}")
    print(f"楠炲啿褰?       : {platform.platform()}")

    found = False
    if check_torch():
        found = True
    if check_tf():
        found = True
    if not found:
        print("\n瑜版挸澧犻悳顖氼暔閺冦垺鐥呴張?PyTorch 娑旂喐鐥呴張?TensorFlow閵?)
    print()



