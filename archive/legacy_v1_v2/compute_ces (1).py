import argparse


def compute_ces(q, t_cpu_ms, s_disk_bytes, budget_ms=50.0, size_cap_bytes=4 * 1024 * 1024):
    if t_cpu_ms > budget_ms:
        return 0.0
    time_penalty = 0.20 * (t_cpu_ms / budget_ms)
    size_penalty = 0.10 * min(1.0, s_disk_bytes / size_cap_bytes)
    return q * (1 - time_penalty - size_penalty)


def main():
    ap = argparse.ArgumentParser(description="Compute CES from Q (DUTS-TE), T_CPU_ms, S_disk_bytes")
    ap.add_argument("--q", type=float, required=True, help="Q on DUTS-TE, e.g. 0.6253")
    ap.add_argument("--t_cpu_ms", type=float, required=True, help="from time_official.py")
    ap.add_argument("--s_disk_bytes", type=int, required=True, help="os.path.getsize(weights.pt)")
    args = ap.parse_args()

    ces = compute_ces(args.q, args.t_cpu_ms, args.s_disk_bytes)
    print(f"Q             = {args.q:.4f}")
    print(f"T_CPU_ms      = {args.t_cpu_ms:.3f}  (budget: 50 ms)")
    print(f"S_disk_bytes  = {args.s_disk_bytes}  ({args.s_disk_bytes / (1024*1024):.3f} MiB, cap: 4 MiB)")
    print(f"CES           = {ces:.4f}")


if __name__ == "__main__":
    main()
