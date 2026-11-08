# Guia de Apresentação Prática (15 Minutos) — 4º Integrante

Este documento foi elaborado especificamente para orientar o **quarto integrante do seminário** durante a demonstração prática ao vivo (aproximadamente **15 minutos**, dentro do seminário total de 60 minutos).

Enquanto os três primeiros integrantes já apresentaram:
1. A motivação intuitiva da simulação quântica;
2. A história de Lie, Trotter e Masuo Suzuki;
3. A fundamentação matemática (BCH, limites de erro e complexidade);

**seu papel é transformar a teoria em evidência experimental verificável ao vivo**, conectando as equações matemáticas com o código Python/Qiskit, os circuitos quânticos e os gráficos de convergência.

---

## ⏱️ Cronograma Sugerido (15 Minutos)

| Tempo | Etapa da Demonstração | O que mostrar na tela |
| :--- | :--- | :--- |
| **00:00 – 02:00** | **1. Arquitetura do Projeto e Validação Exata** | Estrutura de pastas e `src/trotter_suzuki/hamiltonian.py` |
| **02:00 – 05:00** | **2. Implementação das 3 Ordens e Circuitos Qiskit** | `src/trotter_suzuki/trotter.py` e `src/trotter_suzuki/circuits.py` |
| **05:00 – 07:00** | **3. Execução no Terminal e Testes Automatizados** | Rodar `python main.py` e `pytest` ao vivo |
| **07:00 – 12:00** | **4. Demonstração Interativa no Streamlit** | Interface `streamlit run app.py` no navegador |
| **12:00 – 15:00** | **5. Interpretação Científica dos Gráficos e Conclusões** | Aba de gráficos / figuras em `results/figures/` |

---

## 1. Como Explicar a Arquitetura do Projeto (Minutos 0 a 2)

Comece destacando que o projeto não é um script monolítico, mas um pacote científico modular e testado:

- `src/trotter_suzuki/hamiltonian.py`: Define as matrizes de Pauli $I, X, Y, Z$, o Hamiltoniano $H = X + Z$, verifica que $[X, Z] = -2iY \neq 0$ e implementa **duas** soluções exatas independentes (via `scipy.linalg.expm` e via fórmula analítica fechada).
- `src/trotter_suzuki/trotter.py`: Implementa puramente em álgebra linear (NumPy) os passos $S_1(\Delta t)$, $S_2(\Delta t)$ e $S_4(\Delta t)$, além das métricas de erro espectral e fidelidade.
- `src/trotter_suzuki/circuits.py`: Traduz as exponenciais em portas nativas $R_X$ e $R_Z$ do **Qiskit**, extrai a matriz unitária do circuito e analisa o custo computacional.
- `src/trotter_suzuki/experiments.py` e `visualization.py`: Executam as 63 configurações experimentais e geram os gráficos em alta resolução.

---

## 2. Quais Arquivos e Funções Mostrar Primeiro (Minutos 2 a 5)

Abra apenas **três trechos curtos de código** no editor (não percorra arquivos inteiros):

### A. Em `src/trotter_suzuki/hamiltonian.py`:
Mostre as funções `exact_evolution_numeric(t)` e `exact_evolution_analytic(t)`:
```python
def exact_evolution_numeric(t: float) -> np.ndarray:
    h_matrix = get_hamiltonian()
    return expm(-1.0j * h_matrix * float(t))

def exact_evolution_analytic(t: float) -> np.ndarray:
    t_val = float(t)
    sqrt2 = np.sqrt(2.0)
    omega = sqrt2 * t_val
    return np.cos(omega) * PAULI_I - 1.0j * (np.sin(omega) / sqrt2) * (PAULI_X + PAULI_Z)
```
> **O que explicar:** *"Como $\{X, Z\} = 0$ e $X^2 = Z^2 = I$, temos $H^2 = (X+Z)^2 = 2I$. Isso nos permite validar a exponencial numérica `expm` contra uma solução analítica exata com erro na casa de $10^{-16}$."*

### B. Em `src/trotter_suzuki/trotter.py`:
Mostre as três funções de passo elementar (`lie_trotter_step_1`, `suzuki_step_2`, `suzuki_step_4`):
```python
SUZUKI_4_P: float = float(1.0 / (4.0 - 4.0 ** (1.0 / 3.0)))

def lie_trotter_step_1(dt: float) -> np.ndarray:
    return exp_ix(dt) @ exp_iz(dt)

def suzuki_step_2(dt: float) -> np.ndarray:
    half_wx = exp_ix(float(dt) / 2.0)
    full_wz = exp_iz(float(dt))
    return half_wx @ full_wz @ half_wx

def suzuki_step_4(dt: float) -> np.ndarray:
    dt_val = float(dt)
    s2_p = suzuki_step_2(SUZUKI_4_P * dt_val)
    s2_p_sq = s2_p @ s2_p
    s2_central = suzuki_step_2((1.0 - 4.0 * SUZUKI_4_P) * dt_val)
    return s2_p_sq @ s2_central @ s2_p_sq
```
> **O que explicar:** *"Reparem na elegância da construção recursiva de Suzuki para a 4ª ordem: `suzuki_step_4` chama `suzuki_step_2` cinco vezes. O subpasso central tem peso $1 - 4p \approx -0.658 < 0$, ou seja, evolui ligeiramente 'para trás' no tempo para cancelar exatamente o comutador de 3ª ordem na série de Baker–Campbell–Hausdorff."*

### C. Em `src/trotter_suzuki/circuits.py`:
Mostre o loop principal dentro de `build_trotter_circuit`:
```python
        if order == 1:
            # S_1(dt) = exp(-i X dt) @ exp(-i Z dt)
            # Fator mais à direita (Z) atua primeiro no circuito:
            qc.rz(2.0 * dt, 0)
            qc.rx(2.0 * dt, 0)
```
> **O que explicar (Ponto Crítico de Engenharia Quântica):**
> 1. *"Pela definição das portas de rotação $R_P(\theta) = \exp(-i \frac{\theta}{2} P)$, um passo $\exp(-i P \Delta t)$ requer o ângulo dobrado $2\Delta t$."*
> 2. *"Em álgebra linear, no produto matricial $S_1(\Delta t) = e^{-iX\Delta t} e^{-iZ\Delta t}$ aplicado a $|\psi_0\rangle$, a matriz da **direita** ($e^{-iZ\Delta t}$) multiplica o vetor primeiro. Já no diagrama de circuito do Qiskit, a leitura temporal vai da **esquerda para a direita**. Por isso, adicionamos `qc.rz(2*dt, 0)` primeiro e `qc.rx(2*dt, 0)` depois. Nossos testes provam que a matriz do circuito Qiskit coincide com a matriz NumPy até $10^{-15}$."*

---

## 3. Como Executar uma Simulação ao Vivo (Minutos 5 a 12)

### Passo 1: Terminal (2 minutos)
No terminal, execute:
```bash
pytest
```
Mostre que todos os **131 testes automatizados** passam em poucos segundos, validando não-comutatividade, unitariedade, equivalência matriz-circuito e taxas de convergência.

Em seguida, caso queira mostrar a reprodutibilidade em lote:
```bash
python main.py
```

### Passo 2: Interface Streamlit (5 minutos)
Abra a interface interativa:
```bash
streamlit run app.py
```
Siga este roteiro visual no navegador:

1. **Configure $t = 1.0$ e $r = 1$ com "Comparar todos os métodos simultaneamente" marcado:**
   - Mostre na tabela comparativa que para $r=1$ passo:
     - **1ª Ordem:** erro espectral $= 0.799$, fidelidade $F = 0.5940$, $P(1) = 0.7081$ (muito distante do exato $0.4878$).
     - **2ª Ordem:** erro espectral $= 0.314$, fidelidade $F = 0.9093$.
     - **4ª Ordem:** erro espectral $= 0.0127$, fidelidade $F = 0.999839$, $P(1) = 0.5005$.
2. **Aumente o controle deslizante de $r$ ao vivo: $r = 1 \to 2 \to 4 \to 16 \to 64$:**
   - Mostre na **Aba 1 (Circuito Quântico)** como o circuito cresce em número de portas para $r=1, 2, 4$.
   - Explique a diferença entre a contagem **canônica** ($2r, 3r, 15r$) e a contagem **com fusão de fronteira** ($2r, 2r+1, 10r+1$), onde rotações $R_X$ vizinhas entre passos consecutivos se somam em uma única porta.
   - Mostre o aviso de proteção visual quando $r > 8$ (ex.: em $r=64$, a 4ª ordem tem $960$ portas).

---

## 4. Como Interpretar os Gráficos Obtidos (Minutos 12 a 15)

Vá para a **Aba 2 (Gráficos de Convergência e Custo)** no Streamlit (ou mostre os arquivos em `results/figures/`):

### Gráfico 1 (`error_vs_steps.png`) — Erro vs. Número de Passos $r$
- Em escala log-log, uma lei de potência $E(r) \propto r^{-k}$ aparece como uma **reta de inclinação $-k$**.
- As três curvas seguem exatamente as retas tracejadas teóricas:
  - **1ª Ordem:** inclinação medida $= -1.0030$ (teórica $-1.0$, erro cai pela metade quando $r$ dobra).
  - **2ª Ordem:** inclinação medida $= -2.0032$ (teórica $-2.0$, erro cai por fator $4$ quando $r$ dobra).
  - **4ª Ordem:** inclinação medida $= -4.0020$ (teórica $-4.0$, erro cai por fator $16$ quando $r$ dobra).

### Gráfico 2 (`infidelity_vs_steps.png`) — Infidelidade $1 - F$ e Precisão Numérica
- Como o erro no estado $|\psi\rangle$ é proporcional a $O(1/r^k)$ e a infidelidade $1 - F = 1 - |\langle\psi_{\text{exato}}|\psi_{\text{aprox}}\rangle|^2 \approx \|\Delta\psi_\perp\|^2$ depende do **quadrado** do erro de amplitude, a infidelidade decai com o **dobro do expoente**:
  - 1ª Ordem: $O(1/r^2)$
  - 2ª Ordem: $O(1/r^4)$
  - 4ª Ordem: $O(1/r^8)$
- **Destaque sobre precisão de ponto flutuante (`float64`):** Para a 4ª ordem em $r = 32$ e $r = 64$, a infidelidade real atinge $\sim 10^{-17}$ e $\sim 10^{-19}$, ultrapassando o *machine epsilon* do padrão IEEE 754 `float64` ($\epsilon \approx 2.22 \times 10^{-16}$). O gráfico mostra como a subtração ingênua `1.0 - F` satura nesse piso, enquanto o cálculo via projeção ortogonal $\|\psi_{\text{aprox}} - \langle\psi_{\text{exato}}|\psi_{\text{aprox}}\rangle\psi_{\text{exato}}\|_2^2$ revela a continuidade exata da lei $O(1/r^8)$.

### Gráfico 3 (`cost_vs_precision.png`) — Custo Computacional vs. Precisão
- Responde à pergunta prática central: *"Vale a pena gastar 15 portas por passo na 4ª ordem em vez de 2 portas por passo na 1ª ordem?"*
- **Sim!** Com apenas $r=1$ passo (**15 portas**), a 4ª ordem atinge erro $1.27 \times 10^{-2}$. Para atingir precisão semelhante na 1ª ordem, são necessários $r=64$ passos (**128 portas**, erro $1.09 \times 10^{-2}$).
- Para qualquer meta de erro abaixo de $10^{-2}$, métodos de ordem superior exigem **muito menos portas totais** e menor profundidade de circuito.

### Gráfico 4 (`measurement_probability.png`) — Probabilidade $P(|1\rangle)$
- À esquerda: a probabilidade converge para o valor analítico exato $\sin^2(\sqrt{2})/2 \approx 0.487841$. Em $r=1$, a 2ª ordem subestima ($0.2067$) e a 1ª ordem superestima ($0.7081$), mas já em $r=4$ a 2ª e a 4ª ordens estão visualmente coladas na linha exata.
- À direita: na evolução contínua no tempo $t \in [0, 2.2]$ com $r=2$ fixo, vemos que o erro de Trotter acumula com o tempo segundo $O(t^{k+1}/r^k)$: a 1ª ordem defasa rapidamente para $t > 1$, enquanto a 4ª ordem acompanha a oscilação de Rabi exata em todo o intervalo.

---

## 5. Conclusões Científicas Justificadas pelos Resultados

Finalize sua fala de 15 minutos sintetizando 4 conclusões rigorosas:

1. **Validação Experimental das Ordens Assintóticas:** Os experimentos confirmaram numericamente as taxas teóricas $O(1/r)$, $O(1/r^2)$ e $O(1/r^4)$ para o erro de operador, e $O(1/r^2)$, $O(1/r^4)$ e $O(1/r^8)$ para a infidelidade de estado.
2. **Superioridade em Custo-Benefício de Ordens Maiores:** Embora cada passo de 2ª e 4ª ordem exija mais exponenciais, a redução drástica no número de passos $r$ necessário para atingir uma precisão alvo resulta em circuitos muito mais rasos.
3. **Equivalência Exata entre Álgebra Linear e Circuitos Quânticos:** Respeitando a convenção de ordenação temporal inversa aos produtos matriciais e o fator $2$ nos ângulos de rotação $R_X(2\Delta t)$ e $R_Z(2\Delta t)$, a simulação por circuitos reproduz as matrizes teóricas com discrepância inferior a $10^{-14}$.
4. **Distinção entre Simulação de 1 Qubit e Sistemas de Muitos Corpos:** Em 1 qubit, um compilador clássico (`transpile` nível 3) pode colapsar qualquer sequência de rotações em no máximo 3 portas de Euler $\text{SU}(2)$. Em sistemas reais de $N$ qubits com termos de interação (como $X_i X_{i+1}$ e $Z_i Z_{i+1}$), cada exponencial exige escadas de portas $\text{CNOT}$, tornando impossível essa compressão clássica e fazendo do método de Trotter–Suzuki uma ferramenta indispensável em computadores quânticos reais.
