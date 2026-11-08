"""
Módulo experiments: Execução sistemática dos experimentos de simulação Hamiltoniana
para avaliação de erro de operador, fidelidade, probabilidade de medição, custo de circuito
e ordem assintótica de convergência.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .circuits import analyze_circuit_cost, build_trotter_circuit, circuit_to_statevector, circuit_to_unitary
from .hamiltonian import (
    evolve_state,
    exact_evolution_analytic,
    exact_evolution_numeric,
    exact_probability_one_analytic,
    measurement_probability_one,
)
from .trotter import (
    phase_invariant_operator_error,
    spectral_norm_error,
    state_fidelity,
    state_infidelity,
    state_infidelity_orthogonal,
    trotter_suzuki_evolution,
)

DEFAULT_R_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_T_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_ORDERS: tuple[int, ...] = (1, 2, 4)

ORDER_NAMES: dict[int, str] = {
    1: "1ª Ordem (Lie–Trotter)",
    2: "2ª Ordem (Suzuki S₂)",
    4: "4ª Ordem (Suzuki S₄)",
}


def run_single_configuration(t: float, r: int, order: int) -> dict[str, object]:
    """
    Executa todos os cálculos (exato, matricial e circuito Qiskit) para uma única
    combinação de tempo `t`, número de passos `r` e ordem `order`.
    """
    t_val = float(t)
    r_int = int(r)
    order_int = int(order)

    # 1. Solução exata (numérica e analítica)
    u_exact_num = exact_evolution_numeric(t_val)
    u_exact_ana = exact_evolution_analytic(t_val)
    exact_method_diff = spectral_norm_error(u_exact_num, u_exact_ana)

    psi_exact = evolve_state(u_exact_num)
    p1_exact_state = measurement_probability_one(psi_exact)
    p1_exact_formula = exact_probability_one_analytic(t_val)

    # 2. Aproximação matricial de Trotter–Suzuki
    u_approx_matrix = trotter_suzuki_evolution(t=t_val, r=r_int, order=order_int)
    psi_approx_matrix = evolve_state(u_approx_matrix)
    p1_approx_matrix = measurement_probability_one(psi_approx_matrix)

    # 3. Construção e simulação via Circuito Quântico Qiskit
    qc = build_trotter_circuit(t=t_val, r=r_int, order=order_int, insert_barriers=False)
    u_approx_circuit = circuit_to_unitary(qc)
    psi_approx_circuit = circuit_to_statevector(qc)
    p1_approx_circuit = measurement_probability_one(psi_approx_circuit)

    # Discrepância numérica entre a matriz e o circuito Qiskit
    matrix_circuit_diff = spectral_norm_error(u_approx_matrix, u_approx_circuit)

    # 4. Métricas de erro de aproximação e fidelidade
    op_error_spectral = spectral_norm_error(u_exact_num, u_approx_matrix)
    op_error_phase_inv = phase_invariant_operator_error(u_exact_num, u_approx_matrix)

    fidelity = state_fidelity(psi_exact, psi_approx_matrix)
    infidelity = state_infidelity(psi_exact, psi_approx_matrix)
    infidelity_ortho = state_infidelity_orthogonal(psi_exact, psi_approx_matrix)

    prob_1_error = abs(p1_exact_formula - p1_approx_matrix)

    # 5. Custo computacional do circuito
    cost = analyze_circuit_cost(t=t_val, r=r_int, order=order_int)

    return {
        "order": order_int,
        "method_name": ORDER_NAMES[order_int],
        "t": t_val,
        "r": r_int,
        "dt": t_val / r_int,
        "operator_error_spectral": op_error_spectral,
        "operator_error_phase_invariant": op_error_phase_inv,
        "fidelity": fidelity,
        "infidelity": infidelity,
        "infidelity_orthogonal": infidelity_ortho,
        "prob_1_exact": p1_exact_formula,
        "prob_1_exact_numeric": p1_exact_state,
        "prob_1_approx": p1_approx_matrix,
        "prob_1_circuit": p1_approx_circuit,
        "prob_1_abs_error": prob_1_error,
        "rx_gates": cost["rx_gates"],
        "rz_gates": cost["rz_gates"],
        "total_gates": cost["total_gates"],
        "depth": cost["depth"],
        "merged_total_gates": cost["merged_total_gates"],
        "merged_depth": cost["merged_depth"],
        "transpiled_total_gates": cost["transpiled_total_gates"],
        "transpiled_depth": cost["transpiled_depth"],
        "matrix_circuit_discrepancy": matrix_circuit_diff,
        "exact_numeric_analytic_diff": exact_method_diff,
    }


def run_all_experiments(
    t_values: Sequence[float] = DEFAULT_T_VALUES,
    r_values: Sequence[int] = DEFAULT_R_VALUES,
    orders: Sequence[int] = DEFAULT_ORDERS,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Executa a grade completa de experimentos e calcula as ordens empíricas locais
    de convergência entre passos sucessivos de r.
    """
    records: list[dict[str, object]] = []
    for t_val in t_values:
        for order in orders:
            prev_r: int | None = None
            prev_err: float | None = None
            for r_val in r_values:
                row = run_single_configuration(t=t_val, r=r_val, order=order)
                curr_err = float(row["operator_error_spectral"])
                if prev_r is not None and prev_err is not None and curr_err > 0.0 and prev_err > 0.0:
                    local_order = -np.log(curr_err / prev_err) / np.log(r_val / prev_r)
                else:
                    local_order = np.nan
                row["empirical_local_order"] = float(local_order) if not np.isnan(local_order) else np.nan
                records.append(row)
                prev_r = r_val
                prev_err = curr_err

    df = pd.DataFrame(records)

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.16e")

    return df


def compute_convergence_summary(df: pd.DataFrame, t: float = 1.0, min_r: int = 4) -> pd.DataFrame:
    """
    Ajusta a inclinação log-log (log(erro) vs log(r)) no regime assintótico (r >= min_r)
    para verificar numericamente as ordens teóricas O(1/r), O(1/r^2) e O(1/r^4).
    """
    sub = df[(np.isclose(df["t"], t)) & (df["r"] >= min_r)].copy()
    summary_rows: list[dict[str, object]] = []

    for order in sorted(sub["order"].unique()):
        grp = sub[sub["order"] == order].sort_values("r")
        log_r = np.log10(grp["r"].to_numpy(dtype=float))
        log_err = np.log10(grp["operator_error_spectral"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(log_r, log_err, 1)

        # Inclinação também para a infidelidade ortogonal (esperada ~ -2 * order)
        infid_vals = grp["infidelity_orthogonal"].to_numpy(dtype=float)
        valid_mask = infid_vals > 0
        if int(np.sum(valid_mask)) >= 2:
            slope_infid, _ = np.polyfit(log_r[valid_mask], np.log10(infid_vals[valid_mask]), 1)
        else:
            slope_infid = np.nan

        summary_rows.append(
            {
                "order": int(order),
                "method_name": ORDER_NAMES[int(order)],
                "t": float(t),
                "theoretical_operator_scaling": f"O(1/r^{int(order)})",
                "theoretical_exponent": -float(order),
                "empirical_exponent_asymptotic": float(slope),
                "asymptotic_prefactor_10_intercept": float(10.0**intercept),
                "theoretical_infidelity_exponent": -2.0 * float(order),
                "empirical_infidelity_exponent": float(slope_infid),
                "error_at_r1": float(
                    df[(np.isclose(df["t"], t)) & (df["order"] == order) & (df["r"] == 1)][
                        "operator_error_spectral"
                    ].iloc[0]
                ),
                "error_at_r64": float(
                    df[(np.isclose(df["t"], t)) & (df["order"] == order) & (df["r"] == 64)][
                        "operator_error_spectral"
                    ].iloc[0]
                ),
            }
        )

    return pd.DataFrame(summary_rows)
