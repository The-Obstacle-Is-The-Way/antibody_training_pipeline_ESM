"""Parse alpha sweep results from Hydra multirun."""

import re
from pathlib import Path

# Path to multirun results
multirun_dir = Path("multirun/2025-11-13/00-07-58")

results: list[tuple[float, float]] = []

# Parse each run (0-8)
for i in range(9):
    run_dir = multirun_dir / str(i)

    # Get alpha from overrides.yaml
    overrides_file = run_dir / ".hydra" / "overrides.yaml"
    with open(overrides_file) as f:
        content = f.read()
        alpha_match = re.search(r"regressor\.alpha=([\d.]+)", content)
        if not alpha_match:
            continue
        alpha = float(alpha_match.group(1))

    # Get Spearman from log
    log_file = run_dir / "train_ginkgo_competition.log"
    spearman: float | None = None
    with open(log_file) as f:
        for line in f:
            if "Overall Spearman:" in line and "core.trainer" in line:
                spearman_match = re.search(r"Overall Spearman: ([\d.]+)", line)
                if spearman_match:
                    spearman = float(spearman_match.group(1))
                break

    if spearman is not None:
        results.append((alpha, spearman))

# Sort by alpha
results.sort()

# Print results
print("\n" + "=" * 60)
print("ALPHA SWEEP RESULTS (ESM-1v + Ridge Regression)")
print("=" * 60)
print(f"{'Alpha':<15} {'Spearman Correlation':<25} {'Change vs α=1.0'}")
print("-" * 60)

baseline_spearman: float | None = None
for alpha, spearman in results:
    if alpha == 1.0:
        baseline_spearman = spearman

for alpha, spearman in results:
    change = ""
    if baseline_spearman is not None and alpha != 1.0:
        diff = spearman - baseline_spearman
        pct = (diff / baseline_spearman) * 100
        change = f"{diff:+.4f} ({pct:+.1f}%)"
    elif alpha == 1.0:
        change = "(baseline)"

    print(f"{alpha:<15.3f} {spearman:<25.4f} {change}")

print("=" * 60)

# Find best
best_alpha, best_spearman = max(results, key=lambda x: x[1])
print(f"\n🏆 BEST RESULT: α = {best_alpha}, Spearman = {best_spearman:.4f}")
if baseline_spearman is not None:
    improvement = best_spearman - baseline_spearman
    improvement_pct = (improvement / baseline_spearman) * 100
    print(f"   Improvement over α=1.0: {improvement:.4f} ({improvement_pct:+.1f}%)")
print("=" * 60)
print()
