"""RTX 3080 20GB 显存全量压测（无数据集，纯代码模拟）

三个阶段：
  1. 上限探测：反复分配并写入显存，直到物理显存真的用尽
  2. 全量压测：把显存填满确定性数据，反复读写 N 秒，再逐块校验数据是否被改写
  3. 训练模拟：用 3D 卷积网络（医学影像常见形态）找最大可训练 batch

注意：Windows 的 WDDM 允许显存"超额分配"（超出部分换页到内存），
所以本脚本不用"分配成功"判断上限，而是以 nvidia-smi 的物理占用为准。

用法：
    python vram_stress.py                # 默认 60 秒压测
    python vram_stress.py --seconds 180  # 压测 3 分钟
    python vram_stress.py --skip-unet    # 跳过第 3 阶段
"""
import argparse
import subprocess
import sys
import threading
import time

import torch

GB = 1024 ** 3


class GpuMonitor(threading.Thread):
    """后台用 nvidia-smi 采样温度、功耗、物理显存占用。"""

    def __init__(self, interval=1.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = False
        self.max_temp = 0
        self.max_power = 0.0
        self.max_mem = 0.0

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                temp, power, mem = out.split(",")
                self.max_temp = max(self.max_temp, int(float(temp)))
                self.max_power = max(self.max_power, float(power))
                self.max_mem = max(self.max_mem, float(mem) / 1024)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag = True


def banner(title):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def fill_pattern(chunk, index, step=1024 * 1024):
    """分片写入确定性数据，避免申请大临时缓冲。"""
    n = chunk.numel()
    for j in range(0, n, step):
        m = min(step, n - j)
        seg = torch.arange(m, dtype=torch.float32, device="cuda") + float(j)
        chunk[j:j + m] = torch.remainder(seg + index * 7919.0, 4096.0)


def checksum(chunk, step=1024 * 1024):
    """按位求和：把 float32 的 bit pattern 当 int32 累加。

    几乎不占额外显存，且对任何比特翻转都敏感。前提是数据里没有 -0.0 / NaN。
    """
    total = 0
    n = chunk.numel()
    for j in range(0, n, step):
        m = min(step, n - j)
        seg = chunk[j:j + m].view(torch.int32)
        total += int(seg.sum(dtype=torch.int64).item())
    return total


def physical_used_gb():
    """从驱动读物理显存占用（不受超额分配影响）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    return float(out) / 1024


def test_max_allocation(chunk_mb=512, reserve_mb=256):
    banner("阶段 1｜显存上限探测（以物理占用为准）")
    torch.cuda.empty_cache()
    base = physical_used_gb()
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"显卡总显存 : {total:.2f} GB")
    print(f"探测前占用 : {base:.2f} GB（桌面/浏览器等）")
    print(f"分块大小   : {chunk_mb} MB，每块分配后立即写入以强制占用物理显存")
    print()

    elems = chunk_mb * 1024 * 1024 // 4
    chunks = []
    peak = base
    try:
        while True:
            c = torch.empty(elems, dtype=torch.float32, device="cuda")
            c.fill_(1.0)              # 真正写进去，逼驱动分配物理页
            chunks.append(c)
            used = physical_used_gb()
            peak = max(peak, used)
            if len(chunks) % 8 == 0:
                print(f"  物理占用 {used:6.2f} GB ...", end="\r", flush=True)
            if used >= total - reserve_mb / 1024:
                print(f"\n  物理显存接近用尽（{used:.2f} GB），停止分配")
                break
    except torch.cuda.OutOfMemoryError:
        print(f"\n  第 {len(chunks) + 1} 块分配失败 —— 到达真实上限")

    torch.cuda.synchronize()
    used = physical_used_gb()
    print(f"\n  物理峰值占用 : {peak:.2f} GB / {total:.2f} GB  ({peak / total * 100:.1f}%)")
    print(f"  可分配给用户 : {peak - base:.2f} GB（扣掉系统占用）")

    for c in chunks:
        del c
    torch.cuda.empty_cache()
    time.sleep(1)
    return peak, total


def test_full_readwrite(seconds=60, chunk_mb=512, ratio=0.88):
    banner(f"阶段 2｜全量读写压测 + 数据完整性校验（{seconds} 秒）")
    torch.cuda.empty_cache()
    time.sleep(1)
    base = physical_used_gb()
    total = torch.cuda.get_device_properties(0).total_memory / GB

    target_used = total * ratio
    n_chunks = max(1, int((target_used - base) * 1024 // chunk_mb))
    chunk_elems = chunk_mb * 1024 * 1024 // 4

    print(f"计划占用   : {base + n_chunks * chunk_mb / 1024:.2f} GB  ({n_chunks} 块 × {chunk_mb} MB)")
    print(f"数据模式   : 每块写入确定性序列，压测后逐块校验")
    print()

    chunks = []
    try:
        for i in range(n_chunks):
            t = torch.empty(chunk_elems, dtype=torch.float32, device="cuda")
            fill_pattern(t, i)
            chunks.append(t)
            if (i + 1) % 8 == 0:
                print(f"  已填充 {physical_used_gb():5.2f} GB ...", end="\r", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"\n  填充到第 {len(chunks)} 块时显存不足，按实际数量继续")

    torch.cuda.synchronize()
    used_phys = physical_used_gb()
    used_torch = torch.cuda.memory_allocated() / GB
    print(f"\n  物理占用   : {used_phys:.2f} GB     PyTorch 统计: {used_torch:.2f} GB")

    print("  正在计算基准校验和 ...")
    baseline = [checksum(c) for c in chunks]
    print(f"  已记录 {len(baseline)} 个基准校验和")

    print(f"\n  开始压测，目标 {seconds} 秒（风扇会转起来）...")
    mon = GpuMonitor()
    mon.start()

    t0 = time.time()
    passes = 0
    bytes_moved = 0
    try:
        while time.time() - t0 < seconds:
            for c in chunks:
                c.add_(0.0)                       # 读 + 写整块
                bytes_moved += c.numel() * 4 * 2
            passes += 1
            elapsed = max(time.time() - t0, 1e-6)
            if passes % 5 == 0:
                bw = bytes_moved / elapsed / 1e9
                print(f"    第 {passes:3d} 轮  {elapsed:5.1f}s  带宽 {bw:6.1f} GB/s   "
                      f"{mon.max_temp}°C  {mon.max_power:.0f}W  {mon.max_mem:.1f}GB",
                      end="\r", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("\n  压测中触发 OOM（显存被其他程序抢占），提前结束")
    finally:
        mon.stop()

    torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-6)
    bw = bytes_moved / elapsed / 1e9
    print(f"\n\n  压测完成   : {passes} 轮，{elapsed:.1f} 秒")
    print(f"  搬运数据   : {bytes_moved / 1e12:.1f} TB")
    print(f"  等效带宽   : {bw:.1f} GB/s  （理论 760 GB/s，{bw / 760 * 100:.0f}%）")
    print(f"  最高温度   : {mon.max_temp} °C    最高功耗: {mon.max_power:.0f} W")
    print(f"  显存峰值   : {mon.max_mem:.2f} GB")

    print("\n  正在逐块校验数据完整性 ...")
    errors = 0
    for i, c in enumerate(chunks):
        after = checksum(c)
        if after != baseline[i]:
            errors += 1
            print(f"    [错误] 第 {i} 块校验和不一致：{baseline[i]} -> {after}")
    if errors == 0:
        print(f"  OK 全部 {len(chunks)} 块数据完全一致，零错误（{used_phys:.2f} GB 全部读写过）")
    else:
        print(f"  !! 发现 {errors} 块数据异常")

    del chunks
    torch.cuda.empty_cache()
    return used_phys, errors


def test_unet_batch():
    banner("阶段 3｜3D 医学影像网络训练模拟：最大 batch")
    torch.cuda.empty_cache()
    # 关键：限制分配器只能用 95% 物理显存，
    # 否则 Windows 会把超出部分换到内存，制造"假成功"。
    torch.cuda.set_per_process_memory_fraction(0.95)
    print("(已限制分配器最多用 95% 物理显存，避免换页造成的假成功)")
    print()

    import torch.nn as nn

    class Small3DUNet(nn.Module):
        """简化版 3D UNet（带解码器），输入 1×128×128×128 的 CT/MRI 体块。"""

        def __init__(self, base=32):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv3d(1, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
                nn.Conv3d(base, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
                nn.MaxPool3d(2),
                nn.Conv3d(base, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
                nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
                nn.MaxPool3d(2),
                nn.Conv3d(base * 2, base * 4, 3, padding=1), nn.InstanceNorm3d(base * 4), nn.LeakyReLU(0.1),
            )
            self.up1 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
            )
            self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv3d(base, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
            )
            self.head = nn.Conv3d(base, 2, 1)

        def forward(self, x):
            x = self.enc(x)
            x = self.dec1(self.up1(x))
            x = self.dec2(self.up2(x))
            return self.head(x)

    def try_batch(bs, patch=128):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = Small3DUNet().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        try:
            x = torch.randn(bs, 1, patch, patch, patch, device="cuda")
            y = torch.randint(0, 2, (bs, patch, patch, patch), device="cuda")
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / GB
            del x, y, out, loss, model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return True, peak
        except torch.cuda.OutOfMemoryError:
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return False, 0.0

    print("固定输入 1×128×128×128 的 3D 体块，逐步加大 batch：")
    print()
    best, peak_at_best = 0, 0.0
    for bs in (1, 2, 3, 4, 6, 8, 12):
        ok, peak = try_batch(bs)
        print(f"  batch={bs:2d}  {'OK ' if ok else 'OOM'}  peak={peak:5.2f} GB")
        if ok:
            best, peak_at_best = bs, peak
        else:
            break

    print(f"\n  最大可训练 batch : {best}  (占用 {peak_at_best:.2f} GB)")
    print(f"  对应体数据量     : {best} × 128³ = {best * 128 ** 3 / 1e6:.1f} M voxel")
    torch.cuda.empty_cache()
    return best


def test_amp_compare():
    """对比 fp32 与 AMP(混合精度) 的最大 batch 和训练速度。"""
    banner("阶段 4｜混合精度 AMP 对比（fp32 vs fp16）")
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

    import torch.nn as nn

    class UNet3D(nn.Module):
        def __init__(self, base=32):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv3d(1, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
                nn.Conv3d(base, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
                nn.MaxPool3d(2),
                nn.Conv3d(base, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
                nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
                nn.MaxPool3d(2),
                nn.Conv3d(base * 2, base * 4, 3, padding=1), nn.InstanceNorm3d(base * 4), nn.LeakyReLU(0.1),
            )
            self.up1 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.InstanceNorm3d(base * 2), nn.LeakyReLU(0.1),
            )
            self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv3d(base, base, 3, padding=1), nn.InstanceNorm3d(base), nn.LeakyReLU(0.1),
            )
            self.head = nn.Conv3d(base, 2, 1)

        def forward(self, x):
            x = self.enc(x)
            x = self.dec1(self.up1(x))
            x = self.dec2(self.up2(x))
            return self.head(x)

    def run_once(bs, amp, patch=128, steps=3):
        """跑 steps 步，返回 (成功?, 峰值GB, 每步秒数)。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = UNet3D().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        try:
            x = torch.randn(bs, 1, patch, patch, patch, device="cuda")
            y = torch.randint(0, 2, (bs, patch, patch, patch), device="cuda")

            def one_step():
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(x)
                    loss = torch.nn.functional.cross_entropy(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            one_step()                       # 预热
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(steps):
                one_step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / steps
            peak = torch.cuda.max_memory_allocated() / GB
            del x, y, model, opt, scaler
            torch.cuda.empty_cache()
            return True, peak, dt
        except torch.cuda.OutOfMemoryError:
            del model, opt, scaler
            torch.cuda.empty_cache()
            return False, 0.0, 0.0

    print("输入 1×128×128×128，各跑 3 步取平均：\n")
    print(f"  {'batch':>6} | {'fp32 峰值':>10} {'fp32 步时':>10} | {'AMP 峰值':>10} {'AMP 步时':>10}")
    print("  " + "-" * 60)

    fp32, amp = {}, {}
    for bs in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        ok32, pk32, dt32 = run_once(bs, amp=False)
        okam, pkam, dtam = run_once(bs, amp=True)
        if ok32:
            fp32[bs] = (pk32, dt32)
        if okam:
            amp[bs] = (pkam, dtam)
        s32 = f"{pk32:6.2f}GB {dt32*1000:7.0f}ms" if ok32 else f"{'OOM':>10} {'-':>10}"
        sam = f"{pkam:6.2f}GB {dtam*1000:7.0f}ms" if okam else f"{'OOM':>10} {'-':>10}"
        print(f"  {bs:>6} | {s32} | {sam}")
        if not ok32 and not okam:
            break

    max32 = max(fp32) if fp32 else 0
    maxamp = max(amp) if amp else 0
    print()
    print(f"  最大 batch : fp32 = {max32},  AMP = {maxamp}")
    if max32 and maxamp:
        print(f"  batch 上限 : 提升 {maxamp / max32:.2f} 倍")
    # 同 batch 下的速度对比
    common = sorted(set(fp32) & set(amp))
    if common:
        bs = common[-1]
        p32, t32 = fp32[bs]
        pam, tam = amp[bs]
        print(f"  同 batch={bs} 对比: fp32 {t32*1000:.0f} ms/步  ->  AMP {tam*1000:.0f} ms/步  "
              f"(提速 {t32 / tam:.2f}×)")

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)
    return max32, maxamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60, help="阶段 2 压测时长（秒）")
    ap.add_argument("--chunk-mb", type=int, default=512, help="分块大小（MB）")
    ap.add_argument("--skip-unet", action="store_true", help="跳过阶段 3")
    ap.add_argument("--only-unet", action="store_true", help="只跑阶段 3")
    ap.add_argument("--amp", action="store_true", help="对比 fp32 与混合精度的最大 batch")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查环境。")
        sys.exit(1)

    p = torch.cuda.get_device_properties(0)
    banner("设备信息")
    print(f"显卡     : {p.name}")
    print(f"显存     : {p.total_memory / GB:.2f} GB")
    print(f"算力     : sm_{p.major}{p.minor}   SM 数量: {p.multi_processor_count}")
    print(f"PyTorch  : {torch.__version__}   CUDA: {torch.version.cuda}")

    t_start = time.time()
    if args.amp:
        test_amp_compare()
        print(f"\n  总耗时: {time.time() - t_start:.0f} 秒")
        return

    if args.only_unet:
        peak, total, used, errors = 0.0, p.total_memory / GB, 0.0, 0
        best_batch = test_unet_batch()
        print(f"\n  3D 训练最大 batch: {best_batch}")
        print(f"  总耗时: {time.time() - t_start:.0f} 秒")
        return

    peak, total = test_max_allocation(args.chunk_mb)
    used, errors = test_full_readwrite(args.seconds, args.chunk_mb)
    best_batch = None if args.skip_unet else test_unet_batch()

    banner("总结果")
    print(f"  显存上限     : 物理峰值 {peak:.2f} GB / {total:.2f} GB")
    print(f"  全量压测     : 占用 {used:.2f} GB，数据错误 {errors} 个")
    if best_batch is not None:
        print(f"  3D 训练 batch: {best_batch}")
    print(f"  总耗时       : {time.time() - t_start:.0f} 秒")
    print()
    if errors == 0 and peak / total > 0.90:
        print("  结论：显存几乎全部可用且读写正确，20 GB 没有虚标。")
    elif errors > 0:
        print("  结论：显存存在数据错误，建议重点关注。")
    else:
        print("  结论：压测无错误，但物理占用没到 90%，可能被其他程序占着。")


if __name__ == "__main__":
    main()
