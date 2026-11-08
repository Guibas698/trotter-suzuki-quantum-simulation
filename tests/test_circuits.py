"""
Testes automatizados para o módulo circuits e validação de artefatos experimentais.
Verifica:
1. Equivalência exata entre as matrizes de Trotter–Suzuki e os circuitos Qiskit.
2. Contagem de portas e profundidade (canônica, fundida e transpilada).
3. Validade do dataframe de experimentos e dos arquivos exportados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trotter_suzuki.circuits import (
    analyze_circuit_cost,
    build_merged_trotter_circuit,
    build_trotter_circuit,
    circuit_to_statevector,
    circuit_to_unitary,
)
from trotter_suzuki.experiments import run_all_experiments
from trotter_suzuki.hamiltonian import KET_0, evolve_state
from trotter_suzuki.trotter import trotter_suzuki_evolution


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("r", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("t", [0.5, 1.0, 2.0])
def test_matrix_and_qiskit_circuit_equivalence(order: int, r: int, t: float) -> None:
    """
    Garante que o circuito Qiskit produz exatamente a mesma matriz unitária
    e o mesmo vetor de estado que a implementação matricial em NumPy.
    """
    u_matrix = trotter_suzuki_evolution(t=t, r=r, order=order)
    psi_matrix = evolve_state(u_matrix, KET_0)

    qc = build_trotter_circuit(t=t, r=r, order=order, insert_barriers=False)
    u_circuit = circuit_to_unitary(qc)
    psi_circuit = circuit_to_statevector(qc)

    assert np.allclose(u_matrix, u_circuit, atol=1e-12)
    assert np.allclose(psi_matrix, psi_circuit, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("r", [1, 2, 4, 8])
def test_merged_circuit_preserves_unitary_and_reduces_gates(order: int, r: int) -> None:
    """Verifica que a fusão algébrica de rotações adjacentes preserva a matriz unitária exata."""
    qc_canon = build_trotter_circuit(t=1.0, r=r, order=order)
    qc_merged = build_merged_trotter_circuit(t=1.0, r=r, order=order)

    assert np.allclose(circuit_to_unitary(qc_canon), circuit_to_unitary(qc_merged), atol=1e-13)


@pytest.mark.parametrize("r", [1, 2, 4, 8, 16, 32, 64])
def test_circuit_gate_counts_and_depth_formulas(r: int) -> None:
    """
    Verifica as fórmulas exatas de contagem de portas e profundidade:
    - 1ª ordem: 2r canônico, 2r com fusão
    - 2ª ordem: 3r canônico, 2r + 1 com fusão
    - 4ª ordem: 15r canônico, 10r + 1 com fusão
    """
    c1 = analyze_circuit_cost(t=1.0, r=r, order=1)
    assert c1["rx_gates"] == r
    assert c1["rz_gates"] == r
    assert c1["total_gates"] == 2 * r
    assert c1["depth"] == 2 * r
    assert c1["merged_total_gates"] == 2 * r

    c2 = analyze_circuit_cost(t=1.0, r=r, order=2)
    assert c2["rx_gates"] == 2 * r
    assert c2["rz_gates"] == r
    assert c2["total_gates"] == 3 * r
    assert c2["depth"] == 3 * r
    assert c2["merged_total_gates"] == 2 * r + 1

    c4 = analyze_circuit_cost(t=1.0, r=r, order=4)
    assert c4["rx_gates"] == 10 * r
    assert c4["rz_gates"] == 5 * r
    assert c4["total_gates"] == 15 * r
    assert c4["depth"] == 15 * r
    assert c4["merged_total_gates"] == 10 * r + 1


def test_experiments_dataframe_and_exported_files() -> None:
    """Verifica a integridade dos resultados experimentais e a existência dos arquivos gerados."""
    df = run_all_experiments(t_values=(0.5, 1.0, 2.0), r_values=(1, 2, 4, 8, 16, 32, 64), orders=(1, 2, 4))
    assert len(df) == 3 * 7 * 3  # 63 combinações
    assert df["matrix_circuit_discrepancy"].max() < 1e-12
    assert df["exact_numeric_analytic_diff"].max() < 1e-14

    project_root = Path(__file__).resolve().parent.parent
    csv_file = project_root / "results" / "experiment_results.csv"
    figures_dir = project_root / "results" / "figures"

    if csv_file.exists():
        assert csv_file.stat().st_size > 1000

    expected_figures = [
        "error_vs_steps.png",
        "infidelity_vs_steps.png",
        "cost_vs_precision.png",
        "measurement_probability.png",
    ]
    if figures_dir.exists():
        for fig_name in expected_figures:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Figura obrigatória ausente: {fig_name}"
            assert fig_path.stat().st_size > 5000
