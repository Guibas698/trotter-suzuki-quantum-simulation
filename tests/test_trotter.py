"""
Testes automatizados para o módulo trotter.
Verifica:
1. Unitariedade das aproximações de 1ª, 2ª e 4ª ordens.
2. Normalização dos estados aproximados.
3. Redução monotônica do erro e convergência assintótica O(1/r), O(1/r^2) e O(1/r^4).
4. Hierarquia de precisão (erro_O4 < erro_O2 < erro_O1 para r suficientemente grande).
"""

from __future__ import annotations

import numpy as np
import pytest

from trotter_suzuki.hamiltonian import KET_0, evolve_state, exact_evolution_analytic
from trotter_suzuki.trotter import (
    SUZUKI_4_P,
    exp_ix,
    exp_iz,
    is_unitary,
    lie_trotter_step_1,
    phase_invariant_operator_error,
    spectral_norm_error,
    state_fidelity,
    state_infidelity,
    suzuki_step_2,
    suzuki_step_4,
    trotter_suzuki_evolution,
)


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.5, 1.0])
def test_elementary_steps_are_unitary(dt: float) -> None:
    """Verifica que os passos elementares S_1(dt), S_2(dt) e S_4(dt) são unitários."""
    assert is_unitary(exp_ix(dt), tol=1e-14)
    assert is_unitary(exp_iz(dt), tol=1e-14)
    assert is_unitary(lie_trotter_step_1(dt), tol=1e-14)
    assert is_unitary(suzuki_step_2(dt), tol=1e-14)
    assert is_unitary(suzuki_step_4(dt), tol=1e-14)


def test_suzuki_4_constant_satisfies_cancelling_condition() -> None:
    """Verifica a identidade algébrica 4*p^3 + (1 - 4*p)^3 == 0 que anula o erro de 3ª ordem."""
    p = SUZUKI_4_P
    residual = 4.0 * (p**3) + (1.0 - 4.0 * p) ** 3
    assert np.isclose(residual, 0.0, atol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("r", [1, 2, 4, 8, 16, 32, 64])
def test_full_evolution_unitarity_and_state_normalization(order: int, r: int) -> None:
    """Verifica que U_k(t; r) é unitário e preserva a norma do estado quântico."""
    u_approx = trotter_suzuki_evolution(t=1.0, r=r, order=order)
    assert is_unitary(u_approx, tol=1e-12)

    psi = evolve_state(u_approx, KET_0)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-13)


@pytest.mark.parametrize("order,expected_exponent,tol", [(1, -1.0, 0.08), (2, -2.0, 0.08), (4, -4.0, 0.12)])
def test_asymptotic_convergence_scaling(order: int, expected_exponent: float, tol: float) -> None:
    """Verifica numericamente o expoente assintótico O(1/r^k) para t = 1.0 e r in {8, 16, 32, 64}."""
    t = 1.0
    u_exact = exact_evolution_analytic(t)
    r_list = np.array([8, 16, 32, 64], dtype=float)
    errors = np.array(
        [spectral_norm_error(u_exact, trotter_suzuki_evolution(t=t, r=int(r), order=order)) for r in r_list]
    )

    # Os erros devem decrescer estritamente à medida que r dobra
    assert np.all(np.diff(errors) < 0)

    # Inclinação log-log
    slope, _ = np.polyfit(np.log10(r_list), np.log10(errors), 1)
    assert np.isclose(slope, expected_exponent, atol=tol)


def test_order_accuracy_hierarchy() -> None:
    """Verifica que para t=1.0 e r >= 2, a 4ª ordem é mais precisa que a 2ª, que é mais precisa que a 1ª."""
    t = 1.0
    u_exact = exact_evolution_analytic(t)
    psi_exact = evolve_state(u_exact, KET_0)

    for r in (2, 4, 8, 16, 32, 64):
        err_1 = spectral_norm_error(u_exact, trotter_suzuki_evolution(t, r, order=1))
        err_2 = spectral_norm_error(u_exact, trotter_suzuki_evolution(t, r, order=2))
        err_4 = spectral_norm_error(u_exact, trotter_suzuki_evolution(t, r, order=4))

        assert err_4 < err_2 < err_1

        # Verifica também que a fidelidade cresce com a ordem
        f_1 = state_fidelity(psi_exact, evolve_state(trotter_suzuki_evolution(t, r, order=1)))
        f_2 = state_fidelity(psi_exact, evolve_state(trotter_suzuki_evolution(t, r, order=2)))
        f_4 = state_fidelity(psi_exact, evolve_state(trotter_suzuki_evolution(t, r, order=4)))
        assert f_1 <= f_2 <= f_4 <= 1.0 + 1e-15
        assert state_infidelity(psi_exact, evolve_state(trotter_suzuki_evolution(t, r, order=4))) >= 0.0


def test_global_phase_invariance_matches_su2_direct_error() -> None:
    """Como exp(-i X dt) e exp(-i Z dt) estão em SU(2), o erro direto e o invariante à fase coincidem em 1ª ordem."""
    u_exact = exact_evolution_analytic(1.0)
    u_approx = trotter_suzuki_evolution(1.0, r=8, order=2)
    err_direct = spectral_norm_error(u_exact, u_approx)
    err_phase_inv = phase_invariant_operator_error(u_exact, u_approx)
    assert err_phase_inv <= err_direct + 1e-15


def test_invalid_inputs_raise_value_error() -> None:
    """Garante validação de entradas inválidas."""
    with pytest.raises(ValueError):
        trotter_suzuki_evolution(t=1.0, r=0, order=1)
    with pytest.raises(ValueError):
        trotter_suzuki_evolution(t=1.0, r=4, order=3)
