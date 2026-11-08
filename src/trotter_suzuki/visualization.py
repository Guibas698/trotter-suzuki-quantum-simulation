"""
Módulo visualization: Geração de gráficos científicos de alta resolução (DPI=300)
para slides acadêmicos, relatórios e interface interativa Streamlit, além de
diagramas de circuitos quânticos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .circuits import build_trotter_circuit
from .hamiltonian import evolve_state, exact_probability_one_analytic, measurement_probability_one
from .trotter import trotter_suzuki_evolution

# Usa backend não-interativo seguro para geração em lote e servidor
matplotlib.use("Agg")

# Limite seguro de passos para renderização visual de diagramas de circuito
MAX_VISUAL_R: int = 8

# Paleta de cores acadêmica com alto contraste (azul escuro, azul médio/ciano, laranja/vermelho)
ORDER_COLORS: dict[int, str] = {
    1: "#1d4ed8",  # Azul Royal (1ª Ordem)
    2: "#0284c7",  # Azul Ciano Escuro (2ª Ordem)
    4: "#dc2626",  # Vermelho Acadêmico (4ª Ordem)
}

ORDER_MARKERS: dict[int, str] = {
    1: "o",
    2: "s",
    4: "^",
}

ORDER_LABELS: dict[int, str] = {
    1: "1ª Ordem: Lie–Trotter $S_1(\\Delta t)$",
    2: "2ª Ordem: Suzuki $S_2(\\Delta t)$",
    4: "4ª Ordem: Suzuki $S_4(\\Delta t)$",
}


def apply_academic_style() -> None:
    """Configura parâmetros globais do Matplotlib para visual acadêmico limpo."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#cbd5e1",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_error_vs_steps(
    df: pd.DataFrame,
    t: float = 1.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Gráfico 1 — Erro de aproximação (norma espectral ||U_exato - U_aprox||_2) vs. número de passos r.
    Inclui curvas de referência assintóticas O(1/r), O(1/r^2) e O(1/r^4).
    """
    apply_academic_style()
    sub = df[np.isclose(df["t"], t)].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    for order in (1, 2, 4):
        grp = sub[sub["order"] == order].sort_values("r")
        r_vals = grp["r"].to_numpy(dtype=float)
        err_vals = grp["operator_error_spectral"].to_numpy(dtype=float)

        ax.loglog(
            r_vals,
            err_vals,
            marker=ORDER_MARKERS[order],
            color=ORDER_COLORS[order],
            linewidth=2.2,
            markersize=7,
            label=ORDER_LABELS[order],
        )

        # Linha guia assintótica O(1/r^k) ancorada em r = 8
        anchor_idx = np.where(r_vals == 8)[0]
        if len(anchor_idx) > 0:
            r0 = r_vals[anchor_idx[0]]
            e0 = err_vals[anchor_idx[0]]
            ref_curve = e0 * (r0 / r_vals) ** order
            ax.loglog(
                r_vals,
                ref_curve,
                linestyle=":",
                color=ORDER_COLORS[order],
                alpha=0.55,
                linewidth=1.4,
                label=f"Referência $O(1/r^{order})$",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Número de Passos de Trotter ($r$)")
    ax.set_ylabel("Erro Espectral $\\|U_{\\mathrm{exato}} - U_{\\mathrm{aprox}}\\|_2$ (adimensional)")
    ax.set_title(f"Convergência do Erro de Operador vs. Passos de Trotter ($t = {t:.1f}$)")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def plot_infidelity_vs_steps(
    df: pd.DataFrame,
    t: float = 1.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Gráfico 2 — Infidelidade (1 - F) vs. número de passos r.
    Destaca também o piso de precisão de máquina float64 (~2.2e-16) para 1 - F direto
    versus a projeção ortogonal contínua O(1/r^{2k}).
    """
    apply_academic_style()
    sub = df[np.isclose(df["t"], t)].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    for order in (1, 2, 4):
        grp = sub[sub["order"] == order].sort_values("r")
        r_vals = grp["r"].to_numpy(dtype=float)
        infid_ortho = grp["infidelity_orthogonal"].to_numpy(dtype=float)
        infid_direct = grp["infidelity"].to_numpy(dtype=float)

        # Plota a infidelidade de alta precisão (projeção ortogonal)
        ax.loglog(
            r_vals,
            infid_ortho,
            marker=ORDER_MARKERS[order],
            color=ORDER_COLORS[order],
            linewidth=2.2,
            markersize=7,
            label=f"{ORDER_LABELS[order]} ($O(1/r^{{{2*order}}})$)",
        )

        # Para a 4ª ordem, mostra onde 1.0 - F em float64 satura no epsilon de máquina
        if order == 4:
            direct_pos = np.maximum(infid_direct, 1e-17)
            ax.loglog(
                r_vals,
                direct_pos,
                marker="x",
                linestyle="--",
                color="#7c2d12",
                alpha=0.75,
                linewidth=1.5,
                label="4ª Ordem: $1 - F$ direto (`float64`)",
            )

    ax.axhline(
        np.finfo(float).eps,
        color="#475569",
        linestyle="-.",
        linewidth=1.2,
        label="Piso de precisão IEEE 754 `float64` ($\\epsilon \\approx 2.22 \\times 10^{-16}$)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Número de Passos de Trotter ($r$)")
    ax.set_ylabel("Infidelidade de Estado $1 - F = 1 - |\\langle\\psi_{\\mathrm{exato}}|\\psi_{\\mathrm{aprox}}\\rangle|^2$")
    ax.set_title(f"Infidelidade do Estado Quântico vs. Passos de Trotter ($t = {t:.1f}$, $|\\psi_0\\rangle = |0\\rangle$)")
    ax.legend(loc="lower left", fontsize=8.8)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def plot_cost_vs_precision(
    df: pd.DataFrame,
    t: float = 1.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Gráfico 3 — Custo computacional (quantidade de portas de rotação no circuito) vs. erro obtido.
    Mostra o trade-off entre número total de portas lógicas e precisão atingida.
    """
    apply_academic_style()
    sub = df[np.isclose(df["t"], t)].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    for order in (1, 2, 4):
        grp = sub[sub["order"] == order].sort_values("r")
        gates = grp["total_gates"].to_numpy(dtype=float)
        err_vals = grp["operator_error_spectral"].to_numpy(dtype=float)
        r_vals = grp["r"].to_numpy(dtype=int)

        ax.loglog(
            gates,
            err_vals,
            marker=ORDER_MARKERS[order],
            color=ORDER_COLORS[order],
            linewidth=2.2,
            markersize=7,
            label=f"{ORDER_LABELS[order]} (Canônico: ${int(gates[0])//int(r_vals[0])}r$ portas)",
        )

        # Anota alguns valores de r nos pontos
        for g_val, e_val, r_val in zip(gates, err_vals, r_vals):
            if r_val in (1, 4, 16, 64):
                ax.annotate(
                    f"$r={r_val}$",
                    (g_val, e_val),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    color=ORDER_COLORS[order],
                )

    ax.set_xlabel("Custo Computacional: Quantidade Total de Portas de Rotação ($R_X + R_Z$)")
    ax.set_ylabel("Erro Espectral $\\|U_{\\mathrm{exato}} - U_{\\mathrm{aprox}}\\|_2$")
    ax.set_title(f"Custo Computacional vs. Precisão da Simulação ($t = {t:.1f}$)")
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def plot_measurement_probability(
    df: pd.DataFrame,
    t: float = 1.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Gráfico 4 — Probabilidade de medição do estado |1>.
    Painel esquerdo: Convergência de P(1) em função do número de passos r para t fixo.
    Painel direito: Evolução temporal contínua P(1)(t) para r = 2 comparada à solução analítica exata.
    """
    apply_academic_style()
    sub = df[np.isclose(df["t"], t)].copy()
    p1_exact_target = exact_probability_one_analytic(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    # Painel A: P(1) vs r
    ax1.axhline(
        p1_exact_target,
        color="#0f172a",
        linestyle="--",
        linewidth=2.0,
        label=f"Exato Analítico ($P(1) = {p1_exact_target:.6f}$)",
    )
    for order in (1, 2, 4):
        grp = sub[sub["order"] == order].sort_values("r")
        r_vals = grp["r"].to_numpy(dtype=float)
        p1_vals = grp["prob_1_approx"].to_numpy(dtype=float)
        ax1.plot(
            r_vals,
            p1_vals,
            marker=ORDER_MARKERS[order],
            color=ORDER_COLORS[order],
            linewidth=2.0,
            markersize=6.5,
            label=ORDER_LABELS[order],
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Número de Passos de Trotter ($r$)")
    ax1.set_ylabel("Probabilidade de Medição $P(|1\\rangle)$")
    ax1.set_title(f"(a) Convergência de $P(|1\\rangle)$ vs. Passos ($t = {t:.1f}$)")
    ax1.legend(loc="best", fontsize=8.8)

    # Painel B: Dinâmica temporal P(1) vs tempo t com r = 2 passos fixos
    t_grid = np.linspace(0.0, 2.2, 160)
    p1_exact_curve = [exact_probability_one_analytic(tv) for tv in t_grid]
    ax2.plot(
        t_grid,
        p1_exact_curve,
        color="#0f172a",
        linestyle="-",
        linewidth=2.4,
        label="Exato: $\\sin^2(\\sqrt{2}t)/2$",
    )

    r_demo = 2
    for order in (1, 2, 4):
        p1_approx_curve = [
            measurement_probability_one(evolve_state(trotter_suzuki_evolution(tv, r=r_demo, order=order)))
            for tv in t_grid
        ]
        ax2.plot(
            t_grid,
            p1_approx_curve,
            linestyle="--" if order == 1 else ("-." if order == 2 else ":"),
            color=ORDER_COLORS[order],
            linewidth=2.0,
            label=f"{ORDER_LABELS[order]} ($r={r_demo}$)",
        )

    ax2.set_ylim(-0.02, 0.78)
    ax2.set_xlabel("Tempo de Evolução $t$ (unidades $\\hbar = 1$)")
    ax2.set_ylabel("Probabilidade de Medição $P(|1\\rangle)$")
    ax2.set_title(f"(b) Oscilação de Rabi em $P(|1\\rangle)$ no Tempo ($r = {r_demo}$ fixo)")
    ax2.legend(loc="upper right", fontsize=8.8)

    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def plot_multi_time_comparison(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Gráfico complementar: Comparação do erro espectral para diferentes intervalos de tempo
    t in {0.5, 1.0, 2.0}, demonstrando o escalonamento O(t^{k+1} / r^k).
    """
    apply_academic_style()
    t_values = sorted(df["t"].unique())
    fig, axes = plt.subplots(1, len(t_values), figsize=(14.0, 4.6), dpi=300, sharey=True)

    if len(t_values) == 1:
        axes = [axes]

    for ax, t_val in zip(axes, t_values):
        sub = df[np.isclose(df["t"], t_val)]
        for order in (1, 2, 4):
            grp = sub[sub["order"] == order].sort_values("r")
            ax.loglog(
                grp["r"],
                grp["operator_error_spectral"],
                marker=ORDER_MARKERS[order],
                color=ORDER_COLORS[order],
                linewidth=2.0,
                markersize=6,
                label=ORDER_LABELS[order],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f"Tempo $t = {t_val:.1f}$")
        ax.set_xlabel("Passos de Trotter ($r$)")
        if ax is axes[0]:
            ax.set_ylabel("Erro Espectral $\\|U_{\\mathrm{exato}} - U_{\\mathrm{aprox}}\\|_2$")
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Dependência do Erro Global com o Intervalo de Tempo Simulado ($t \\in \\{0.5, 1.0, 2.0\\}$)", y=1.02, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def render_circuit_diagram(
    t: float,
    r: int,
    order: int,
    save_path: str | Path | None = None,
    insert_barriers: bool = True,
) -> plt.Figure:
    """
    Renderiza o diagrama visual do circuito quântico usando Matplotlib via Qiskit.
    Protege contra renderização de circuitos excessivamente grandes (r > MAX_VISUAL_R).
    """
    if int(r) > MAX_VISUAL_R:
        raise ValueError(
            f"Renderização visual desativada para r={r} (> {MAX_VISUAL_R}) "
            "para evitar diagramas excessivamente extensos."
        )

    qc = build_trotter_circuit(t=t, r=r, order=order, insert_barriers=insert_barriers)
    fig = qc.draw(output="mpl", fold=12, style="iqp")
    fig.suptitle(
        f"Circuito Trotter–Suzuki | Ordem {order} | t = {t:.2f}, r = {r} passo(s) "
        f"({qc.size()} portas, profundidade {qc.depth()})",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def generate_all_figures(df: pd.DataFrame, figures_dir: str | Path) -> dict[str, Path]:
    """
    Gera todos os gráficos obrigatórios e diagramas de circuito para exemplos pequenos (r = 1, 2, 4)
    e retorna um dicionário mapeando o identificador da figura para o caminho salvo em disco.
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 4 gráficos obrigatórios + 1 gráfico comparativo multi-tempo
    p1 = out_dir / "error_vs_steps.png"
    fig1 = plot_error_vs_steps(df, t=1.0, save_path=p1)
    plt.close(fig1)
    generated["error_vs_steps"] = p1

    p2 = out_dir / "infidelity_vs_steps.png"
    fig2 = plot_infidelity_vs_steps(df, t=1.0, save_path=p2)
    plt.close(fig2)
    generated["infidelity_vs_steps"] = p2

    p3 = out_dir / "cost_vs_precision.png"
    fig3 = plot_cost_vs_precision(df, t=1.0, save_path=p3)
    plt.close(fig3)
    generated["cost_vs_precision"] = p3

    p4 = out_dir / "measurement_probability.png"
    fig4 = plot_measurement_probability(df, t=1.0, save_path=p4)
    plt.close(fig4)
    generated["measurement_probability"] = p4

    p5 = out_dir / "multi_time_comparison.png"
    fig5 = plot_multi_time_comparison(df, save_path=p5)
    plt.close(fig5)
    generated["multi_time_comparison"] = p5

    # Diagramas de circuito para exemplos pequenos r in {1, 2, 4}
    for order in (1, 2, 4):
        for r_val in (1, 2, 4):
            c_path = out_dir / f"circuit_order{order}_r{r_val}.png"
            c_fig = render_circuit_diagram(t=1.0, r=r_val, order=order, save_path=c_path)
            plt.close(c_fig)
            generated[f"circuit_order{order}_r{r_val}"] = c_path

    return generated
