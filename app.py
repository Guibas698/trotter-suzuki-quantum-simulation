"""
Interface Interativa Streamlit para Demonstração Acadêmica ao Vivo:
"Simulação Quântica com Trotter–Suzuki"

Como executar:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotter_suzuki.circuits import build_trotter_circuit
from trotter_suzuki.experiments import (
    DEFAULT_R_VALUES,
    ORDER_NAMES,
    compute_convergence_summary,
    run_all_experiments,
    run_single_configuration,
)
from trotter_suzuki.hamiltonian import (
    check_non_commutativity,
    exact_evolution_analytic,
    exact_evolution_numeric,
    exact_probability_one_analytic,
)
from trotter_suzuki.trotter import trotter_suzuki_evolution
from trotter_suzuki.visualization import (
    MAX_VISUAL_R,
    plot_cost_vs_precision,
    plot_error_vs_steps,
    plot_infidelity_vs_steps,
    plot_measurement_probability,
    render_circuit_diagram,
)

st.set_page_config(
    page_title="Simulação Quântica com Trotter–Suzuki",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS acadêmico minimalista (fundo claro, azul institucional, alto contraste)
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
        border-left: 6px solid #1d4ed8;
        padding: 1.2rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: #1e3a8a;
        margin: 0 0 0.35rem 0;
        font-size: 2.0rem;
    }
    .main-header p {
        color: #334155;
        margin: 0;
        font-size: 1.02rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #1d4ed8;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cabeçalho Principal
st.markdown(
    """
    <div class="main-header">
        <h1>Simulação Quântica com Trotter–Suzuki</h1>
        <p>
            Demonstração experimental interativa da decomposição da evolução temporal
            <b>U(t) = e<sup>&minus;iHt</sup></b> para o Hamiltoniano não-comutativo de 1 qubit
            <b>H = X + Z</b> (&hbar; = 1) utilizando circuitos quânticos no Qiskit.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Barra Lateral: Controles Interativos
st.sidebar.header("⚙️ Parâmetros da Simulação")

order_options = {
    1: "1ª Ordem — Lie–Trotter S₁(Δt)",
    2: "2ª Ordem — Suzuki Simétrico S₂(Δt)",
    4: "4ª Ordem — Suzuki Recursivo S₄(Δt)",
}

selected_order = st.sidebar.selectbox(
    "Ordem do Método de Trotter–Suzuki:",
    options=[1, 2, 4],
    format_func=lambda k: order_options[k],
    index=1,
)

t_mode = st.sidebar.radio(
    "Modo de Seleção do Tempo (t):",
    options=["Tempos Padrão do Experimento", "Controle Contínuo"],
    index=0,
)

if t_mode == "Tempos Padrão do Experimento":
    selected_t = float(
        st.sidebar.select_slider(
            "Tempo de evolução (t):",
            options=[0.5, 1.0, 1.5, 2.0],
            value=1.0,
        )
    )
else:
    selected_t = float(
        st.sidebar.slider(
            "Tempo de evolução (t):",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.05,
        )
    )

selected_r = int(
    st.sidebar.select_slider(
        "Quantidade de passos de Trotter (r):",
        options=list(DEFAULT_R_VALUES),
        value=4,
    )
)

compare_all = st.sidebar.checkbox(
    "Comparar todos os métodos simultaneamente (1ª, 2ª e 4ª ordens)",
    value=True,
)

show_barriers = st.sidebar.checkbox(
    "Exibir barreiras visuais entre passos no circuito",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Nota Científica:** Esta aplicação executa a simulação clássica exata de circuitos "
    "quânticos via álgebra linear e `qiskit.quantum_info`, permitindo isolar o erro "
    "matemático de Trotterização sem ruído de hardware físico."
)


@st.cache_data(show_spinner=False)
def get_cached_experiments_for_t(t_val: float) -> pd.DataFrame:
    return run_all_experiments(t_values=(t_val,), r_values=DEFAULT_R_VALUES, orders=(1, 2, 4))


# Executa cálculo da configuração selecionada e da grade de passos para o tempo t
single_res = run_single_configuration(t=selected_t, r=selected_r, order=selected_order)
df_current_t = get_cached_experiments_for_t(selected_t)

# Painel de Métricas da Configuração Selecionada
st.subheader(
    f"📊 Resultados para {ORDER_NAMES[selected_order]} | Tempo $t = {selected_t:.2f}$ | Passos $r = {selected_r}$ "
    f"($\\Delta t = {selected_t / selected_r:.4f}$)"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Erro de Aproximação ||U - U_aprox||₂",
    f"{single_res['operator_error_spectral']:.3e}",
    help="Norma espectral da diferença entre o operador exato exp(-iHt) e a aproximação de Trotter–Suzuki.",
)
col2.metric(
    "Fidelidade de Estado (F)",
    f"{single_res['fidelity']:.10f}",
    delta=f"1 - F = {single_res['infidelity_orthogonal']:.2e}",
    delta_color="off",
    help="Fidelidade F = |<psi_exato|psi_aprox>|^2 partindo de |0>.",
)
col3.metric(
    "Probabilidade P(|1⟩)",
    f"{single_res['prob_1_approx']:.6f}",
    delta=f"Exata: {single_res['prob_1_exact']:.6f} (|ΔP| = {single_res['prob_1_abs_error']:.2e})",
    delta_color="off",
    help="Probabilidade de medir o estado |1> ao final da evolução.",
)
col4.metric(
    "Portas de Rotação (Canônico)",
    f"{single_res['total_gates']} portas",
    delta=f"{single_res['rx_gates']} Rx + {single_res['rz_gates']} Rz (Fundido: {single_res['merged_total_gates']})",
    delta_color="off",
    help="Contagem de portas no circuito original vs. circuito com fusão de rotações Rx adjacentes.",
)
col5.metric(
    "Profundidade do Circuito",
    f"{single_res['depth']}",
    delta=f"Transpilado (SU2): {single_res['transpiled_depth']}",
    delta_color="off",
    help="Profundidade lógica do circuito original vs. após síntese de Euler de 1 qubit no Qiskit.",
)

# Se Comparação Simultânea estiver ativa, exibe tabela comparativa direta entre 1ª, 2ª e 4ª ordem para (t, r)
if compare_all:
    st.markdown(f"#### 🔍 Comparação Direta das Três Ordens em $t = {selected_t:.2f}$ e $r = {selected_r}$")
    comp_rows = df_current_t[df_current_t["r"] == selected_r].copy()
    display_comp = pd.DataFrame(
        {
            "Método": comp_rows["method_name"],
            "Passo Δt": comp_rows["dt"].map(lambda x: f"{x:.4f}"),
            "Erro Espectral ||ΔU||₂": comp_rows["operator_error_spectral"].map(lambda x: f"{x:.4e}"),
            "Fidelidade (F)": comp_rows["fidelity"].map(lambda x: f"{x:.12f}"),
            "Infidelidade (1 - F)": comp_rows["infidelity_orthogonal"].map(lambda x: f"{x:.4e}"),
            "P(|1⟩) Aproximada": comp_rows["prob_1_approx"].map(lambda x: f"{x:.8f}"),
            "P(|1⟩) Exata": comp_rows["prob_1_exact"].map(lambda x: f"{x:.8f}"),
            "Erro |ΔP(1)|": comp_rows["prob_1_abs_error"].map(lambda x: f"{x:.4e}"),
            "Portas (Canônico)": comp_rows["total_gates"],
            "Portas (Com Fusão)": comp_rows["merged_total_gates"],
            "Profundidade": comp_rows["depth"],
            "Discrepância Matriz–Circuito": comp_rows["matrix_circuit_discrepancy"].map(lambda x: f"{x:.2e}"),
        }
    )
    st.dataframe(display_comp, width="stretch", hide_index=True)

# Abas de Visualização Científica
tab_circuit, tab_conv, tab_exact, tab_table = st.tabs(
    [
        "🔌 Circuito Quântico (Qiskit)",
        "📈 Gráficos de Convergência e Custo",
        "🎯 Comparação com a Solução Exata",
        "📋 Tabela Completa de Experimentos",
    ]
)

with tab_circuit:
    st.markdown("### Diagrama e Estrutura do Circuito Quântico")
    st.markdown(
        r"""
        As exponenciais individuais são mapeadas exatamente em portas nativas de rotação de 1 qubit:
        $$e^{-i X \Delta t} = R_X(2\Delta t), \qquad e^{-i Z \Delta t} = R_Z(2\Delta t)$$
        **Atenção à ordenação temporal:** Em álgebra linear, no produto matricial $S_1(\Delta t) = e^{-iX\Delta t} e^{-iZ\Delta t}$,
        o operador mais à direita ($e^{-iZ\Delta t}$) atua **primeiro** sobre o estado $|\psi_0\rangle$. Por isso, no circuito Qiskit,
        a porta $R_Z(2\Delta t)$ precede a porta $R_X(2\Delta t)$.
        """
    )

    if selected_r <= MAX_VISUAL_R:
        if compare_all and selected_r <= 4:
            st.markdown(f"#### Comparação Visual dos Circuitos para $r = {selected_r}$ passo(s)")
            for ord_idx in (1, 2, 4):
                with st.expander(f"Visualizar Circuito — {ORDER_NAMES[ord_idx]} (r = {selected_r})", expanded=(ord_idx == selected_order)):
                    fig_c = render_circuit_diagram(
                        t=selected_t,
                        r=selected_r,
                        order=ord_idx,
                        insert_barriers=show_barriers,
                    )
                    st.pyplot(fig_c)
                    plt.close(fig_c)
        else:
            fig_c = render_circuit_diagram(
                t=selected_t,
                r=selected_r,
                order=selected_order,
                insert_barriers=show_barriers,
            )
            st.pyplot(fig_c)
            plt.close(fig_c)
    else:
        st.warning(
            f"⚠️ Para $r = {selected_r}$ passos, o circuito possui **{single_res['total_gates']} portas** "
            f"(profundidade {single_res['depth']}). Conforme boas práticas de visualização, "
            f"a renderização gráfica completa é limitada a $r \\le {MAX_VISUAL_R}$ para preservar a legibilidade. "
            "Reduza $r$ para 1, 2, 4 ou 8 no controle lateral para inspecionar o diagrama de portas."
        )
        qc_preview = build_trotter_circuit(t=selected_t, r=1, order=selected_order, insert_barriers=False)
        st.markdown(f"Abaixo, exibimos a célula unitária de **1 passo ($r=1$)** que se repete ${selected_r}$ vezes:")
        fig_preview = render_circuit_diagram(t=selected_t / selected_r, r=1, order=selected_order, insert_barriers=False)
        st.pyplot(fig_preview)
        plt.close(fig_preview)

with tab_conv:
    st.markdown("### Análise Experimental de Convergência e Custo Computacional")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### 1. Erro Espectral vs. Número de Passos ($r$)")
        fig1 = plot_error_vs_steps(df_current_t, t=selected_t)
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("#### 3. Custo Computacional (Portas) vs. Precisão")
        fig3 = plot_cost_vs_precision(df_current_t, t=selected_t)
        st.pyplot(fig3)
        plt.close(fig3)

    with g_col2:
        st.markdown("#### 2. Infidelidade ($1 - F$) vs. Número de Passos ($r$)")
        fig2 = plot_infidelity_vs_steps(df_current_t, t=selected_t)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("#### 4. Probabilidade de Medir $|1\\rangle$")
        fig4 = plot_measurement_probability(df_current_t, t=selected_t)
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("#### 📐 Expoentes Empíricos de Convergência Assintótica ($r \\ge 4$)")
    conv_summary = compute_convergence_summary(df_current_t, t=selected_t, min_r=4)
    st.dataframe(
        pd.DataFrame(
            {
                "Método": conv_summary["method_name"],
                "Escalonamento Teórico": conv_summary["theoretical_operator_scaling"],
                "Expoente Teórico (Erro Operador)": conv_summary["theoretical_exponent"].map(lambda x: f"{x:.2f}"),
                "Expoente Empírico Medido (log-log)": conv_summary["empirical_exponent_asymptotic"].map(
                    lambda x: f"{x:.4f}"
                ),
                "Expoente Teórico (Infidelidade 1-F)": conv_summary["theoretical_infidelity_exponent"].map(
                    lambda x: f"{x:.2f}"
                ),
                "Expoente Empírico (Infidelidade)": conv_summary["empirical_infidelity_exponent"].map(
                    lambda x: f"{x:.4f}"
                ),
                "Erro em r=1": conv_summary["error_at_r1"].map(lambda x: f"{x:.4e}"),
                "Erro em r=64": conv_summary["error_at_r64"].map(lambda x: f"{x:.4e}"),
            }
        ),
        width="stretch",
        hide_index=True,
    )

with tab_exact:
    st.markdown("### Validação Matemática: Solução Exata Analítica vs. Numérica vs. Trotter–Suzuki")
    comm = check_non_commutativity()
    st.latex(
        r"H = X + Z = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \qquad "
        r"[X, Z] = XZ - ZX = -2iY = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix} \neq 0"
    )
    st.latex(
        r"U_{\text{exato}}(t) = e^{-iHt} = \cos(\sqrt{2}t)I - i\frac{\sin(\sqrt{2}t)}{\sqrt{2}}(X + Z), \qquad "
        r"P_{\text{exato}}(|1\rangle) = \frac{\sin^2(\sqrt{2}t)}{2}"
    )

    u_num = exact_evolution_numeric(selected_t)
    u_ana = exact_evolution_analytic(selected_t)
    u_trot = trotter_suzuki_evolution(selected_t, selected_r, selected_order)

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("**Matriz Exata Analítica $U_{\\text{ana}}(t)$**")
        st.code(np.array2string(u_ana, precision=6, suppress_small=True))
    with m_col2:
        st.markdown("**Matriz Exata Numérica `scipy.linalg.expm`**")
        st.code(np.array2string(u_num, precision=6, suppress_small=True))
    with m_col3:
        st.markdown(f"**Matriz Trotter–Suzuki $U_{selected_order}(t; r={selected_r})$**")
        st.code(np.array2string(u_trot, precision=6, suppress_small=True))

    st.success(
        f"✔ Diferença entre Solução Exata Analítica e `scipy.linalg.expm`: "
        f"**{np.linalg.norm(u_num - u_ana, ord=2):.3e}** | "
        f"Diferença entre Matriz Trotter–Suzuki e Circuito Qiskit: "
        f"**{single_res['matrix_circuit_discrepancy']:.3e}**"
    )

with tab_table:
    st.markdown(f"### Tabela Completa de Resultados para $t = {selected_t:.2f}$ ($r \\in \\{{1, 2, 4, 8, 16, 32, 64\\}}$)")
    formatted_table = pd.DataFrame(
        {
            "Ordem": df_current_t["order"],
            "Método": df_current_t["method_name"],
            "r": df_current_t["r"],
            "Δt": df_current_t["dt"].map(lambda x: f"{x:.5f}"),
            "Erro Espectral ||ΔU||₂": df_current_t["operator_error_spectral"].map(lambda x: f"{x:.6e}"),
            "Fidelidade F": df_current_t["fidelity"].map(lambda x: f"{x:.12f}"),
            "Infidelidade (1-F direto)": df_current_t["infidelity"].map(lambda x: f"{x:.4e}"),
            "Infidelidade (Ortogonal)": df_current_t["infidelity_orthogonal"].map(lambda x: f"{x:.4e}"),
            "P(|1⟩) Aprox": df_current_t["prob_1_approx"].map(lambda x: f"{x:.8f}"),
            "P(|1⟩) Exato": df_current_t["prob_1_exact"].map(lambda x: f"{x:.8f}"),
            "Portas (Canônico)": df_current_t["total_gates"],
            "Portas (Fundido)": df_current_t["merged_total_gates"],
            "Profundidade": df_current_t["depth"],
            "Ordem Local Empírica": df_current_t["empirical_local_order"].map(
                lambda x: "—" if pd.isna(x) else f"{x:.3f}"
            ),
        }
    )
    st.dataframe(formatted_table, width="stretch", hide_index=True)

    csv_bytes = df_current_t.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar Resultados em CSV",
        data=csv_bytes,
        file_name=f"trotter_suzuki_results_t{selected_t:.2f}.csv",
        mime="text/csv",
    )
