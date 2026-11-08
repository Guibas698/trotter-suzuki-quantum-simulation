"""
Testes automatizados para o módulo hamiltonian.
Verifica:
1. Corretude algébrica das matrizes de Pauli (I, X, Y, Z).
2. Não-comutatividade [X, Z] = -2i Y != 0.
3. Equivalência entre solução exata numérica (scipy.linalg.expm) e analítica fechada.
4. Normalização do estado evoluído e probabilidade analítica P(|1>) = sin^2(sqrt(2)*t)/2.
"""

from __future__ import annotations

import numpy as np
import pytest

from trotter_suzuki.hamiltonian import (
    KET_0,
    KET_1,
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    check_non_commutativity,
    commutator,
    evolve_state,
    exact_evolution_analytic,
    exact_evolution_numeric,
    exact_probability_one_analytic,
    get_hamiltonian,
    measurement_probability_one,
)
from trotter_suzuki.trotter import is_unitary


def test_pauli_matrices_properties() -> None:
    """Verifica involução (P^2 = I), hermiticidade, unitariedade e traço nulo das matrizes de Pauli."""
    for pauli in (PAULI_X, PAULI_Y, PAULI_Z):
        # Hermiticidade: P == P^dagger
        assert np.allclose(pauli, pauli.conj().T, atol=1e-15)
        # Involução: P^2 == I
        assert np.allclose(pauli @ pauli, PAULI_I, atol=1e-15)
        # Unitariedade
        assert is_unitary(pauli, tol=1e-15)
        # Traço nulo
        assert np.isclose(np.trace(pauli), 0.0, atol=1e-15)

    # Ação sobre a base computacional |0> e |1>
    assert np.allclose(PAULI_X @ KET_0, KET_1, atol=1e-15)
    assert np.allclose(PAULI_X @ KET_1, KET_0, atol=1e-15)
    assert np.allclose(PAULI_Z @ KET_0, KET_0, atol=1e-15)
    assert np.allclose(PAULI_Z @ KET_1, -KET_1, atol=1e-15)


def test_non_commutativity_x_and_z() -> None:
    """Verifica que [X, Z] != 0 e que [X, Z] = -2i Y."""
    comm = commutator(PAULI_X, PAULI_Z)
    expected = np.array([[0.0, -2.0], [2.0, 0.0]], dtype=np.complex128)
    assert np.allclose(comm, expected, atol=1e-15)
    assert np.allclose(comm, -2.0j * PAULI_Y, atol=1e-15)

    info = check_non_commutativity()
    assert info["is_non_commutative"] is True
    assert info["matches_minus_2i_Y"] is True
    assert np.isclose(float(info["spectral_norm"]), 2.0, atol=1e-14)


def test_hamiltonian_properties() -> None:
    """Verifica que H = X + Z é Hermitiano e satisfaz H^2 = 2I."""
    h_mat = get_hamiltonian()
    assert np.allclose(h_mat, h_mat.conj().T, atol=1e-15)
    assert np.allclose(h_mat @ h_mat, 2.0 * PAULI_I, atol=1e-15)


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, np.pi])
def test_exact_evolution_numeric_vs_analytic(t: float) -> None:
    """Valida a equivalência entre scipy.linalg.expm e a fórmula analítica fechada."""
    u_num = exact_evolution_numeric(t)
    u_ana = exact_evolution_analytic(t)

    assert is_unitary(u_num, tol=1e-14)
    assert is_unitary(u_ana, tol=1e-14)
    assert np.allclose(u_num, u_ana, atol=1e-14)


@pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 1.0, 2.0, 3.14159])
def test_exact_state_normalization_and_probability_one(t: float) -> None:
    """Verifica que o estado exato evoluído é normalizado e reproduz P(1) = sin^2(sqrt(2)t)/2."""
    u_exact = exact_evolution_numeric(t)
    psi_t = evolve_state(u_exact, KET_0)

    # Normalização ||psi(t)||_2 == 1
    norm = np.linalg.norm(psi_t)
    assert np.isclose(norm, 1.0, atol=1e-14)

    # Probabilidade P(|1>)
    p1_from_state = measurement_probability_one(psi_t)
    p1_analytic = exact_probability_one_analytic(t)
    assert np.isclose(p1_from_state, p1_analytic, atol=1e-14)
    assert 0.0 <= p1_from_state <= 0.5 + 1e-15
