"""
Módulo hamiltonian: Definição do sistema quântico de 1 qubit (H = X + Z) e solução exata.

Unidades naturais adotadas: hbar = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

# Matrizes de Pauli e Identidade 2x2 (complex128)
PAULI_I: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
PAULI_X: np.ndarray = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y: np.ndarray = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z: np.ndarray = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

# Estados da base computacional
KET_0: np.ndarray = np.array([1.0, 0.0], dtype=np.complex128)
KET_1: np.ndarray = np.array([0.0, 1.0], dtype=np.complex128)


def get_hamiltonian() -> np.ndarray:
    """
    Retorna o Hamiltoniano do sistema de 1 qubit:
        H = X + Z = [[1, 1], [1, -1]]
    """
    return PAULI_X + PAULI_Z


def commutator(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    """
    Calcula o comutador [A, B] = A B - B A entre dois operadores matriciais.
    """
    return op_a @ op_b - op_b @ op_a


def check_non_commutativity(tol: float = 1e-12) -> dict[str, object]:
    """
    Verifica analítica e numericamente que [X, Z] != 0.

    Para as matrizes de Pauli X e Z:
        [X, Z] = XZ - ZX = -2i Y = [[0, -2], [2, 0]]
    """
    comm_xz = commutator(PAULI_X, PAULI_Z)
    expected = -2.0j * PAULI_Y
    norm_comm = float(np.linalg.norm(comm_xz, ord=2))
    is_non_zero = norm_comm > tol
    matches_identity = bool(np.allclose(comm_xz, expected, atol=tol))

    return {
        "commutator_matrix": comm_xz,
        "spectral_norm": norm_comm,
        "is_non_commutative": is_non_zero,
        "matches_minus_2i_Y": matches_identity,
    }


def exact_evolution_numeric(t: float) -> np.ndarray:
    """
    Calcula o operador de evolução temporal exato numericamente usando exponenciação matricial (Padé):
        U(t) = exp(-i H t)
    via scipy.linalg.expm.
    """
    h_matrix = get_hamiltonian()
    return expm(-1.0j * h_matrix * float(t))


def exact_evolution_analytic(t: float) -> np.ndarray:
    """
    Calcula o operador de evolução temporal exato pela fórmula fechada analítica:
        U(t) = cos(sqrt(2) * t) * I - i * (sin(sqrt(2) * t) / sqrt(2)) * (X + Z)

    Derivação:
        Como {X, Z} = XZ + ZX = 0 e X^2 = Z^2 = I, temos H^2 = (X + Z)^2 = 2I.
        Logo, a série de Taylor de exp(-i H t) separa-se nos termos pares e ímpares com frequência sqrt(2).
    """
    t_val = float(t)
    sqrt2 = np.sqrt(2.0)
    omega = sqrt2 * t_val
    return np.cos(omega) * PAULI_I - 1.0j * (np.sin(omega) / sqrt2) * (PAULI_X + PAULI_Z)


def evolve_state(unitary: np.ndarray, initial_state: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica um operador unitário U ao estado inicial |psi_0> (padrão |0>).
    """
    psi_0 = KET_0 if initial_state is None else np.asarray(initial_state, dtype=np.complex128)
    return unitary @ psi_0


def measurement_probability_one(state: np.ndarray) -> float:
    """
    Calcula a probabilidade de medir o estado |1> dado um vetor de estado |psi>:
        P(1) = |<1|psi>|^2
    """
    amplitude_one = np.vdot(KET_1, state)
    return float(np.abs(amplitude_one) ** 2)


def exact_probability_one_analytic(t: float) -> float:
    """
    Expressão analítica fechada para a probabilidade de medir |1> partindo de |0>:
        P(1) = sin^2(sqrt(2) * t) / 2
    """
    t_val = float(t)
    return float((np.sin(np.sqrt(2.0) * t_val) ** 2) / 2.0)
