"""
Módulo circuits: Construção de circuitos quânticos no Qiskit para as fórmulas de
Trotter–Suzuki (1ª, 2ª e 4ª ordens), simulação de operador/estado e análise de custo.

Identidades fundamentais utilizadas:
    exp(-i * X * dt) = R_X(2 * dt)
    exp(-i * Z * dt) = R_Z(2 * dt)

Convenção de ordenação Qiskit vs. Álgebra Linear:
    Em álgebra linear, no produto matricial A @ B aplicado a |psi>, o operador B (à direita)
    atua primeiro sobre o estado |psi>, e A (à esquerda) atua depois.
    No Qiskit, as portas adicionadas primeiro ao circuito atuam primeiro no tempo.
    Portanto, para implementar S_1(dt) = exp(-i X dt) @ exp(-i Z dt), adicionamos primeiro
    R_Z(2 * dt) e em seguida R_X(2 * dt).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from .trotter import SUZUKI_4_P


def _append_suzuki_2_substep(qc: QuantumCircuit, sub_dt: float) -> None:
    """
    Adiciona um subpasso simétrico de 2ª ordem S_2(sub_dt) ao circuito:
        S_2(sub_dt) = exp(-i X sub_dt / 2) exp(-i Z sub_dt) exp(-i X sub_dt / 2)
                    = R_X(sub_dt) R_Z(2 * sub_dt) R_X(sub_dt)
    """
    qc.rx(float(sub_dt), 0)
    qc.rz(2.0 * float(sub_dt), 0)
    qc.rx(float(sub_dt), 0)


def build_trotter_circuit(
    t: float,
    r: int,
    order: int = 1,
    insert_barriers: bool = False,
) -> QuantumCircuit:
    """
    Constrói o circuito quântico canônico (sem fusão algébrica de portas adjacentes)
    correspondente à aproximação de Trotter–Suzuki de ordem 1, 2 ou 4.

    Parâmetros:
        t: Tempo total de evolução.
        r: Número de passos de Trotter (r >= 1).
        order: Ordem da decomposição (1, 2 ou 4).
        insert_barriers: Se True, insere barreiras visuais entre os passos de Trotter.

    Retorna:
        QuantumCircuit de 1 qubit.
    """
    if int(r) < 1:
        raise ValueError(f"O número de passos 'r' deve ser >= 1. Recebido: {r}")
    if order not in (1, 2, 4):
        raise ValueError(f"Ordem inválida ({order}). Use 1, 2 ou 4.")

    r_int = int(r)
    dt = float(t) / r_int
    qc = QuantumCircuit(1, name=f"Trotter_O{order}_r{r_int}")

    for step in range(r_int):
        if order == 1:
            # S_1(dt) = exp(-i X dt) @ exp(-i Z dt)
            # Fator mais à direita (Z) atua primeiro no circuito:
            qc.rz(2.0 * dt, 0)
            qc.rx(2.0 * dt, 0)
        elif order == 2:
            # S_2(dt) = exp(-i X dt/2) @ exp(-i Z dt) @ exp(-i X dt/2)
            _append_suzuki_2_substep(qc, dt)
        elif order == 4:
            # S_4(dt) = S_2(p*dt)^2 @ S_2((1 - 4p)*dt) @ S_2(p*dt)^2
            p_dt = SUZUKI_4_P * dt
            central_dt = (1.0 - 4.0 * SUZUKI_4_P) * dt
            for sub_dt in (p_dt, p_dt, central_dt, p_dt, p_dt):
                _append_suzuki_2_substep(qc, sub_dt)

        if insert_barriers and step < r_int - 1:
            qc.barrier()

    return qc


def build_merged_trotter_circuit(t: float, r: int, order: int = 1) -> QuantumCircuit:
    """
    Constrói o circuito com fusão algébrica de rotações R_X adjacentes na fronteira
    entre passos/subpassos consecutivos (mantendo a estrutura de Trotter sem colapsar
    eixos distintos X e Z).

    Contagem resultante:
        - 1ª ordem: 2r portas (nenhuma porta adjacente compartilha o mesmo eixo).
        - 2ª ordem: 2r + 1 portas (redução de 3r para 2r + 1 pela fusão R_X(dt)R_X(dt) = R_X(2dt)).
        - 4ª ordem: 10r + 1 portas (redução de 15r para 10r + 1 pela fusão nos subpassos e fronteiras).
    """
    canonical_qc = build_trotter_circuit(t=t, r=r, order=order, insert_barriers=False)
    merged_ops: list[tuple[str, float]] = []

    for instruction in canonical_qc.data:
        gate_name = instruction.operation.name
        if gate_name not in ("rx", "rz"):
            continue
        angle = float(instruction.operation.params[0])
        if merged_ops and merged_ops[-1][0] == gate_name:
            prev_name, prev_angle = merged_ops[-1]
            merged_ops[-1] = (prev_name, prev_angle + angle)
        else:
            merged_ops.append((gate_name, angle))

    merged_qc = QuantumCircuit(1, name=f"Trotter_Merged_O{order}_r{int(r)}")
    for gate_name, angle in merged_ops:
        if gate_name == "rx":
            merged_qc.rx(angle, 0)
        elif gate_name == "rz":
            merged_qc.rz(angle, 0)

    return merged_qc


def circuit_to_unitary(qc: QuantumCircuit) -> np.ndarray:
    """
    Extrai a matriz unitária exata 2x2 representada pelo QuantumCircuit usando qiskit.quantum_info.Operator.
    """
    return np.asarray(Operator(qc).data, dtype=np.complex128)


def circuit_to_statevector(qc: QuantumCircuit, initial_label: str = "0") -> np.ndarray:
    """
    Evolui um estado inicial (padrão |0>) através do circuito quântico usando qiskit.quantum_info.Statevector.
    """
    sv = Statevector.from_label(initial_label)
    evolved_sv = sv.evolve(qc)
    return np.asarray(evolved_sv.data, dtype=np.complex128)


def analyze_circuit_cost(t: float, r: int, order: int = 1) -> dict[str, int]:
    """
    Analisa detalhadamente o custo computacional do circuito quântico em três níveis:
    1. Circuito Original (Canônico): cada exponencial e^{-i P alpha} traduzida diretamente em R_P(2 alpha).
    2. Circuito com Fusão de Fronteira: rotações R_X consecutivas fundidas em uma única rotação.
    3. Circuito Transpilado (Otimização Nível 3 do Qiskit na base {rx, rz}): síntese de Euler SU(2)
       para 1 qubit (no máximo 3 rotações, pois qualquer produto de rotações em 1 qubit equivale a uma
       única matriz SU(2) decomponível como R_Z R_X R_Z).
    """
    qc_canonical = build_trotter_circuit(t=t, r=r, order=order, insert_barriers=False)
    ops_canonical = qc_canonical.count_ops()
    rx_count = int(ops_canonical.get("rx", 0))
    rz_count = int(ops_canonical.get("rz", 0))
    canonical_total = rx_count + rz_count
    canonical_depth = int(qc_canonical.depth())

    qc_merged = build_merged_trotter_circuit(t=t, r=r, order=order)
    ops_merged = qc_merged.count_ops()
    merged_rx = int(ops_merged.get("rx", 0))
    merged_rz = int(ops_merged.get("rz", 0))
    merged_total = merged_rx + merged_rz
    merged_depth = int(qc_merged.depth())

    qc_transpiled = transpile(
        qc_canonical,
        basis_gates=["rx", "rz"],
        optimization_level=3,
        seed_transpiler=42,
    )
    ops_transpiled = qc_transpiled.count_ops()
    transpiled_total = int(ops_transpiled.get("rx", 0)) + int(ops_transpiled.get("rz", 0))
    transpiled_depth = int(qc_transpiled.depth())

    return {
        "order": int(order),
        "r": int(r),
        "rx_gates": rx_count,
        "rz_gates": rz_count,
        "total_gates": canonical_total,
        "depth": canonical_depth,
        "merged_rx_gates": merged_rx,
        "merged_rz_gates": merged_rz,
        "merged_total_gates": merged_total,
        "merged_depth": merged_depth,
        "transpiled_total_gates": transpiled_total,
        "transpiled_depth": transpiled_depth,
    }
