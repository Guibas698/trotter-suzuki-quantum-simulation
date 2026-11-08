"""
Módulo trotter: Implementação matricial das fórmulas de produto de Lie–Trotter (1ª ordem)
e Suzuki (2ª e 4ª ordens), além de métricas de erro de operadores e fidelidade de estados.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import PAULI_I, PAULI_X, PAULI_Z

# Constante fractal de Suzuki para a decomposição de 4ª ordem:
# p = 1 / (4 - 4^(1/3)) ≈ 0.4144907717943757
SUZUKI_4_P: float = float(1.0 / (4.0 - 4.0 ** (1.0 / 3.0)))


def exp_ix(dt: float) -> np.ndarray:
    """
    Calcula o operador unitário exponencial de Pauli X:
        exp(-i * X * dt) = cos(dt) * I - i * sin(dt) * X
    Equivalente à porta de rotação R_X(2 * dt).
    """
    angle = float(dt)
    return np.cos(angle) * PAULI_I - 1.0j * np.sin(angle) * PAULI_X


def exp_iz(dt: float) -> np.ndarray:
    """
    Calcula o operador unitário exponencial de Pauli Z:
        exp(-i * Z * dt) = cos(dt) * I - i * sin(dt) * Z
    Equivalente à porta de rotação R_Z(2 * dt).
    """
    angle = float(dt)
    return np.cos(angle) * PAULI_I - 1.0j * np.sin(angle) * PAULI_Z


def lie_trotter_step_1(dt: float) -> np.ndarray:
    """
    Passo elementar de Lie–Trotter de 1ª ordem:
        S_1(dt) = exp(-i X dt) @ exp(-i Z dt)

    Nota sobre ordem de atuação:
        O fator mais à direita, exp(-i Z dt), atua primeiro sobre o vetor de estado |psi>,
        seguido por exp(-i X dt).
    """
    return exp_ix(dt) @ exp_iz(dt)


def suzuki_step_2(dt: float) -> np.ndarray:
    """
    Passo elementar simétrico de Suzuki de 2ª ordem (Strang splitting):
        S_2(dt) = exp(-i X dt / 2) @ exp(-i Z dt) @ exp(-i X dt / 2)
    """
    half_wx = exp_ix(float(dt) / 2.0)
    full_wz = exp_iz(float(dt))
    return half_wx @ full_wz @ half_wx


def suzuki_step_4(dt: float) -> np.ndarray:
    """
    Passo elementar fractal de Suzuki de 4ª ordem via composição recursiva de S_2:
        S_4(dt) = S_2(p * dt)^2 @ S_2((1 - 4p) * dt) @ S_2(p * dt)^2
    onde p = 1 / (4 - 4^(1/3)).

    Nota: O passo central possui peso negativo (1 - 4p ≈ -0.65796), o que cancela os
    termos de erro de 3ª ordem na série de Baker–Campbell–Hausdorff (BCH).
    """
    dt_val = float(dt)
    s2_p = suzuki_step_2(SUZUKI_4_P * dt_val)
    s2_p_sq = s2_p @ s2_p
    s2_central = suzuki_step_2((1.0 - 4.0 * SUZUKI_4_P) * dt_val)
    return s2_p_sq @ s2_central @ s2_p_sq


def trotter_suzuki_evolution(t: float, r: int, order: int = 1) -> np.ndarray:
    """
    Calcula o operador de evolução temporal aproximado U_k(t; r) = [S_k(t / r)]^r.

    Parâmetros:
        t: Tempo total de evolução.
        r: Número de passos de Trotter (r >= 1).
        order: Ordem do método (1, 2 ou 4).

    Retorna:
        Matriz unitária 2x2 (np.ndarray complex128).
    """
    if int(r) < 1:
        raise ValueError(f"O número de passos 'r' deve ser um inteiro >= 1. Recebido: {r}")
    r_int = int(r)
    dt = float(t) / r_int

    if order == 1:
        step_op = lie_trotter_step_1(dt)
    elif order == 2:
        step_op = suzuki_step_2(dt)
    elif order == 4:
        step_op = suzuki_step_4(dt)
    else:
        raise ValueError(f"Ordem inválida ({order}). As ordens suportadas são 1, 2 e 4.")

    return np.linalg.matrix_power(step_op, r_int)


def spectral_norm_error(u_exact: np.ndarray, u_approx: np.ndarray) -> float:
    """
    Calcula o erro de aproximação entre dois operadores usando a norma espectral (norma-2 induzida):
        E = ||U_exato - U_aprox||_2 = sigma_max(U_exato - U_aprox)

    Observação física importante:
        A comparação direta ||U_exato - U_aprox||_2 é sensível à fase global (se U_aprox = e^{i phi} U_exato,
        a norma direta não seria nula para phi != 0, embora representem a mesma evolução física).
        Neste problema (geradores de Pauli de traço nulo), tanto U_exato quanto U_aprox pertencem a SU(2)
        com determinante 1 e fase global consistente. Ainda assim, disponibilizamos também
        phase_invariant_operator_error() para comparação invariante à fase global.
    """
    return float(np.linalg.norm(u_exact - u_approx, ord=2))


def phase_invariant_operator_error(u_exact: np.ndarray, u_approx: np.ndarray) -> float:
    """
    Calcula a distância em norma espectral minimizada sobre todas as fases globais:
        min_{phi} ||U_exato - e^{i phi} U_aprox||_2
    alinhando a fase pelo argumento de Tr(U_exato^dagger @ U_aprox).
    """
    overlap = np.trace(u_exact.conj().T @ u_approx)
    if np.abs(overlap) > 1e-15:
        phase = overlap / np.abs(overlap)
        u_aligned = u_approx * np.conj(phase)
    else:
        u_aligned = u_approx
    return float(np.linalg.norm(u_exact - u_aligned, ord=2))


def state_fidelity(psi_exact: np.ndarray, psi_approx: np.ndarray) -> float:
    """
    Calcula a fidelidade quântica entre dois estados puros:
        F = |<psi_exato | psi_aprox>|^2 / (||psi_exato||^2 ||psi_aprox||^2)
    A normalização explícita evita deriva de arredondamento IEEE 754 acumulada após
    múltiplas potências matriciais quando 1 - F < 1e-14.
    """
    u = psi_exact / np.linalg.norm(psi_exact)
    v = psi_approx / np.linalg.norm(psi_approx)
    inner = np.vdot(u, v)
    return float(min(1.0, np.abs(inner) ** 2))


def state_infidelity(psi_exact: np.ndarray, psi_approx: np.ndarray) -> float:
    """
    Calcula a infidelidade direta:
        1 - F = 1 - |<psi_exato | psi_aprox>|^2

    Nota sobre precisão numérica:
        Quando F se aproxima de 1 com diferença menor que o epsilon de máquina IEEE 754 float64
        (~2.22e-16), a subtração direta `1.0 - F` atinge o piso de cancelamento numérico (~1e-16 ou 0.0).
    """
    f_val = state_fidelity(psi_exact, psi_approx)
    return float(max(0.0, 1.0 - f_val))


def state_infidelity_orthogonal(psi_exact: np.ndarray, psi_approx: np.ndarray) -> float:
    """
    Calcula a infidelidade 1 - F = sin^2(theta_FS) evitando cancelamento catastrófico
    via projeção ortogonal sobre estados normalizados:
        1 - |<u|v>|^2 = ||v - <u|v> u||_2^2
    Permite observar o decaimento O(1/r^{2k}) mesmo abaixo do epsilon de máquina (1e-16).
    """
    u = psi_exact / np.linalg.norm(psi_exact)
    v = psi_approx / np.linalg.norm(psi_approx)
    inner = np.vdot(u, v)
    perp = v - inner * u
    return float(np.vdot(perp, perp).real)


def is_unitary(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Verifica se uma matriz U é unitária: U^dagger @ U == I e U @ U^dagger == I.
    """
    n = matrix.shape[0]
    ident = np.eye(n, dtype=np.complex128)
    return bool(
        np.allclose(matrix.conj().T @ matrix, ident, atol=tol)
        and np.allclose(matrix @ matrix.conj().T, ident, atol=tol)
    )
