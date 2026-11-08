"""
Pacote trotter_suzuki: Simulação Quântica da Evolução Temporal via Fórmulas de Produto de Trotter–Suzuki.

Módulos:
- hamiltonian: Definição das matrizes de Pauli, Hamiltoniano H = X + Z e solução exata (analítica e numérica).
- trotter: Decomposições matriciais de Lie–Trotter (1ª ordem) e Suzuki (2ª e 4ª ordens) e métricas de erro.
- circuits: Construção, simulação e análise de custo de circuitos quânticos no Qiskit.
- experiments: Execução sistemática dos experimentos obrigatórios e análise de convergência.
- visualization: Geração de gráficos científicos e diagramas de circuitos para publicação e slides.
"""

from .hamiltonian import (
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    KET_0,
    KET_1,
    get_hamiltonian,
    commutator,
    check_non_commutativity,
    exact_evolution_numeric,
    exact_evolution_analytic,
    evolve_state,
    measurement_probability_one,
    exact_probability_one_analytic,
)
from .trotter import (
    SUZUKI_4_P,
    exp_ix,
    exp_iz,
    lie_trotter_step_1,
    suzuki_step_2,
    suzuki_step_4,
    trotter_suzuki_evolution,
    spectral_norm_error,
    phase_invariant_operator_error,
    state_fidelity,
    state_infidelity,
    is_unitary,
)
from .circuits import (
    build_trotter_circuit,
    circuit_to_unitary,
    circuit_to_statevector,
    analyze_circuit_cost,
)

__version__ = "1.0.0"
__all__ = [
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "KET_0",
    "KET_1",
    "get_hamiltonian",
    "commutator",
    "check_non_commutativity",
    "exact_evolution_numeric",
    "exact_evolution_analytic",
    "evolve_state",
    "measurement_probability_one",
    "exact_probability_one_analytic",
    "SUZUKI_4_P",
    "exp_ix",
    "exp_iz",
    "lie_trotter_step_1",
    "suzuki_step_2",
    "suzuki_step_4",
    "trotter_suzuki_evolution",
    "spectral_norm_error",
    "phase_invariant_operator_error",
    "state_fidelity",
    "state_infidelity",
    "is_unitary",
    "build_trotter_circuit",
    "circuit_to_unitary",
    "circuit_to_statevector",
    "analyze_circuit_cost",
]
