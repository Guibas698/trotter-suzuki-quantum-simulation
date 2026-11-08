"""
Script principal de execução dos experimentos científicos e geração dos gráficos.

Como executar:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garante que o pacote em src/ seja importável sem necessidade de pip install prévio
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotter_suzuki.experiments import compute_convergence_summary, run_all_experiments
from trotter_suzuki.hamiltonian import check_non_commutativity, exact_probability_one_analytic
from trotter_suzuki.visualization import generate_all_figures


def main() -> None:
    print("=" * 88)
    print(" SIMULAÇÃO QUÂNTICA DE TROTTER–SUZUKI | H = X + Z (hbar = 1)")
    print("=" * 88)

    # 1. Verificação de não-comutatividade [X, Z] != 0
    comm_info = check_non_commutativity()
    print("\n[1] Verificação Algébrica do Hamiltoniano:")
    print(f"    ||[X, Z]||_2 = {comm_info['spectral_norm']:.6f} (Não-comutativo: {comm_info['is_non_commutative']})")
    print(f"    [X, Z] == -2i Y confirmado: {comm_info['matches_minus_2i_Y']}")
    print(f"    Probabilidade analítica exata P(|1>) em t = 1.0: {exact_probability_one_analytic(1.0):.10f}")

    # 2. Execução dos experimentos obrigatórios
    results_dir = PROJECT_ROOT / "results"
    csv_path = results_dir / "experiment_results.csv"
    figures_dir = results_dir / "figures"

    print("\n[2] Executando bateria de experimentos para t in {0.5, 1.0, 2.0} e r in {1, 2, 4, 8, 16, 32, 64}...")
    df = run_all_experiments(output_csv_path=csv_path)
    print(f"    -> {len(df)} configurações avaliadas e salvas em: {csv_path}")

    # 3. Sumário para t = 1.0
    print("\n[3] Resultados Experimentais Principais (t = 1.0):")
    sub_t1 = df[df["t"] == 1.0][
        [
            "order",
            "r",
            "operator_error_spectral",
            "fidelity",
            "infidelity",
            "prob_1_approx",
            "total_gates",
            "depth",
            "empirical_local_order",
        ]
    ]
    print(sub_t1.to_string(index=False))

    # 4. Análise de convergência assintótica
    print("\n[4] Verificação da Ordem Assintótica de Convergência (t = 1.0, r >= 4):")
    conv_df = compute_convergence_summary(df, t=1.0, min_r=4)
    print(
        conv_df[
            [
                "order",
                "theoretical_operator_scaling",
                "theoretical_exponent",
                "empirical_exponent_asymptotic",
                "error_at_r1",
                "error_at_r64",
            ]
        ].to_string(index=False)
    )

    # 5. Geração de todos os gráficos e diagramas de circuitos
    print("\n[5] Gerando gráficos científicos e diagramas de circuitos em results/figures/...")
    generated_files = generate_all_figures(df, figures_dir=figures_dir)
    for name, path in generated_files.items():
        print(f"    -> [{name}]: {path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 88)
    print(" Execução concluída com sucesso!")
    print("=" * 88)


if __name__ == "__main__":
    main()
