"""
Compute the CES (Cost-Efficient Segmentation) score.

CES = Q * (1 - 0.20*(T_CPU/50) - 0.10*min(1, S_disk/4MiB))

If T_CPU > 50 ms, CES = 0.

Usage:
  python compute_ces.py --q_duts_te 0.7870 --t_cpu_ms 25.0 --s_disk_bytes 249631

Or pass no args and it will prompt you.
"""
import argparse


def compute_ces(q_score: float, t_cpu_ms: float, s_disk_bytes: int) -> float:
    if t_cpu_ms > 50.0:
        return 0.0
    four_mib = 4 * 1024 * 1024
    time_penalty = 0.20 * (t_cpu_ms / 50.0)
    size_penalty = 0.10 * min(1.0, s_disk_bytes / four_mib)
    return float(q_score * (1.0 - time_penalty - size_penalty))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_duts_te", type=float, default=None,
                    help="Q score on DUTS-TE (from evaluate.py output)")
    ap.add_argument("--t_cpu_ms", type=float, default=None,
                    help="Average T_CPU in ms (from time_official_1.py output)")
    ap.add_argument("--s_disk_bytes", type=int, default=249631,
                    help="Size of weights file in bytes (default: 249631 for weights_1.pt)")
    args = ap.parse_args()

    q = args.q_duts_te
    t = args.t_cpu_ms

    if q is None:
        q = float(input("Enter Q score on DUTS-TE: "))
    if t is None:
        t = float(input("Enter T_CPU_ms (from time_official_1.py): "))

    s = args.s_disk_bytes

    ces = compute_ces(q, t, s)

    print(f"\n{'='*50}")
    print(f"  Q (DUTS-TE):       {q:.4f}")
    print(f"  T_CPU (ms):        {t:.2f}")
    print(f"  S_disk (bytes):    {s}  ({s/1024:.1f} KiB)")
    print(f"  Time penalty:      {0.20 * (t / 50.0):.4f}")
    print(f"  Size penalty:      {0.10 * min(1.0, s / (4 * 1024 * 1024)):.4f}")
    print(f"  CES:               {ces:.4f}")
    print(f"{'='*50}")

    if t > 50.0:
        print("  WARNING: T_CPU > 50 ms → CES = 0 (budget exceeded)")


if __name__ == "__main__":
    main()

