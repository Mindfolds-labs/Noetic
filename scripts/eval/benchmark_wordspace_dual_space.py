from __future__ import annotations

import time

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from pyfolds.leibreg.wordspace import WordSpace, WordSpaceConfig


def run() -> None:
    if torch is None:
        print("[WordSpace benchmark] skipped: torch not installed")
        return
    batch = 1024
    text_dim = 768
    reps = 40
    x = torch.randn(batch, text_dim)

    old = WordSpace(WordSpaceConfig(text_input_dim=text_dim, target_dim=4, hyper_dim=4, monitor_dim=4))
    new = WordSpace(WordSpaceConfig(text_input_dim=text_dim, target_dim=4, hyper_dim=256, monitor_dim=4))

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(reps):
            a = old.project_text(x)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        for _ in range(reps):
            h = new.project_text(x)
            b = new.project_monitor(h)
        t3 = time.perf_counter()

    old_runtime = t1 - t0
    new_runtime = t3 - t2

    old_var = a.var(dim=0).mean().item()
    new_var = h.var(dim=0).mean().item()
    old_norm = a.norm(dim=-1).mean().item()
    new_norm = h.norm(dim=-1).mean().item()
    old_mem_mb = a.numel() * a.element_size() / (1024 * 1024)
    new_mem_mb = (h.numel() * h.element_size() + b.numel() * b.element_size()) / (1024 * 1024)
    old_pairwise = torch.matmul(a[:128], a[:128].transpose(0, 1)).mean().item()
    new_pairwise = torch.matmul(h[:128], h[:128].transpose(0, 1)).mean().item()

    print("[WordSpace benchmark]")
    print(f"old_direct4d_runtime_s={old_runtime:.6f}")
    print(f"new_hyper_plus_monitor_runtime_s={new_runtime:.6f}")
    print(f"old_norm_mean={old_norm:.6f}")
    print(f"new_hyper_norm_mean={new_norm:.6f}")
    print(f"old_var_mean={old_var:.6f}")
    print(f"new_hyper_var_mean={new_var:.6f}")
    print(f"old_memory_mb={old_mem_mb:.4f}")
    print(f"new_memory_mb={new_mem_mb:.4f}")
    print(f"old_pairwise_similarity_mean={old_pairwise:.6f}")
    print(f"new_pairwise_similarity_mean={new_pairwise:.6f}")


if __name__ == "__main__":
    run()
