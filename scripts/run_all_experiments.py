"""Master experimental orchestration runner for AirCast Research.

Sequentially executes:
- Experiment 2 & 3: Feature Engineering Ablation & Marginal Utility
- Experiment 4: Hyperparameter Optimization on Validation Partition
- Experiment 5 & 6: Horizon Lead Time Decay & Seasonal Generalisation
- Phase 6 / Experiment 7: 12-Fold Leave-One-Station-Out Cross-Domain Generalisation
- Phase 8 / Experiment 8: Robustness Evaluation & Stress Testing
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.evaluation.robustness import run_robustness_evaluation
from src.experiments.ablation import run_experiment_2_3
from src.experiments.hyperparameter_tuning import run_experiment_4
from src.experiments.loso_generalisation import run_experiment_7_loso
from src.experiments.temporal_generalisation import run_experiment_5_6


def main():
    print("=" * 80)
    print("AIRCAST RESEARCH: MASTER RESEARCH EXPERIMENTS ORCHESTRATOR")
    print("=" * 80)

    t_global = time.perf_counter()

    # Step 1: Feature Engineering Ablation
    abl_path = "artifacts/metrics/experiment2_3_ablation.json"
    if not os.path.exists(abl_path):
        print("\n>>> STARTING EXPERIMENTS 2 & 3: FEATURE ABLATION ANALYSIS")
        run_experiment_2_3()
    else:
        print(f"\n[Skipping] Experiments 2 & 3 metrics already present at '{abl_path}'.")

    # Step 2: Hyperparameter Optimization
    tune_path = "artifacts/metrics/experiment4_hyperparameter_tuning.json"
    if not os.path.exists(tune_path):
        print("\n>>> STARTING EXPERIMENT 4: HYPERPARAMETER OPTIMIZATION")
        run_experiment_4()
    else:
        print(f"\n[Skipping] Experiment 4 metrics already present at '{tune_path}'.")

    # Step 3: Horizon Degradation & Seasonal Generalisation
    print("\n>>> STARTING EXPERIMENTS 5 & 6: HORIZON DEGRADATION & TEMPORAL GENERALISATION")
    run_experiment_5_6()

    # Step 4: Leave-One-Station-Out Spatial Generalisation
    print("\n>>> STARTING PHASE 6 / EXPERIMENT 7: LEAVE-ONE-STATION-OUT (LOSO)")
    run_experiment_7_loso()

    # Step 5: Robustness Evaluation & Stress Testing
    print("\n>>> STARTING PHASE 8 / EXPERIMENT 8: ROBUSTNESS & STRESS TESTING")
    run_robustness_evaluation()

    total_time = time.perf_counter() - t_global
    print("\n" + "=" * 80)
    print(f"ALL AIRCAST RESEARCH EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time/60.0:.2f} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
