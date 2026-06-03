import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import google.generativeai as genai
import os

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Turbomáquinas CFD & Design Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# FUNÇÕES DE CÁLCULO E FÍSICA
# ==========================================
def calcular_cinematica_iterativa(maq, rho, inputs):
    """
    Solucionador cinemático iterativo para Streamlit.
    Recebe um dicionário de inputs parciais e resolve o resto iterativamente.
    """
    v = inputs.copy()
    changed = True
    iters = 0

    def set_val(k, val):
        nonlocal changed
        if k not in v or v[k] is None or np.isnan(v[k]):
            v[k] = val
            changed = True

    while changed and iters < 100:
        changed = False
        iters += 1

        if v.get('N') is not None: set_val('omega', v['N'] * math.pi / 30.0)
        if v.get('omega') is not None: set_val('N', v['omega'] * 30.0 / math.pi)

        for i in [1, 2]:
            s_i = str(i)
            D = v.get(f'D{s_i}')
            b = v.get(f'b{s_i}')
            U = v.get(f'U{s_i}')
            Cm = v.get(f'Cm{s_i}')
            Cu = v.get(f'Cu{s_i}')
            Wu = v.get(f'Wu{s_i}')
            C = v.get(f'C{s_i}')
            W = v.get(f'W{s_i}')
            alpha = v.get(f'alpha{s_i}')
            beta = v.get(f'beta{s_i}')
            Q = v.get('Q')
            omega = v.get('omega')

            if U is not None and omega is not None and omega != 0: set_val(f'D{s_i}', 2 * U / omega)
            if U is not None and D is not None and D != 0: set_val('omega', 2 * U / D)
            if omega is not None and D is not None: set_val(f'U{s_i}', omega * D / 2)

            if Q is not None and D is not None and b is not None and D * b != 0: set_val(f'Cm{s_i}', Q / (math.pi * D * b))
            if Cm is not None and D is not None and b is not None and Cm != 0: set_val('Q', Cm * math.pi * D * b)
            if Q is not None and Cm is not None and b is not None and Cm * b != 0: set_val(f'D{s_i}', Q / (math.pi * Cm * b))
            if Q is not None and Cm is not None and D is not None and Cm * D != 0: set_val(f'b{s_i}', Q / (math.pi * Cm * D))

            if U is not None and Cu is not None: set_val(f'Wu{s_i}', U - Cu)
            if U is not None and Wu is not None: set_val(f'Cu{s_i}', U - Wu)
            if Cu is not None and Wu is not None: set_val(f'U{s_i}', Cu + Wu)

            if C is not None and Cm is not None: set_val(f'Cu{s_i}', math.sqrt(max(0, C**2 - Cm**2)))
            if C is not None and Cu is not None: set_val(f'Cm{s_i}', math.sqrt(max(0, C**2 - Cu**2)))
            if Cm is not None and Cu is not None: set_val(f'C{s_i}', math.sqrt(Cm**2 + Cu**2))

            if W is not None and Cm is not None: set_val(f'Wu{s_i}', math.sqrt(max(0, W**2 - Cm**2)))
            if W is not None and Wu is not None: set_val(f'Cm{s_i}', math.sqrt(max(0, W**2 - Wu**2)))
            if Cm is not None and Wu is not None: set_val(f'W{s_i}', math.sqrt(Cm**2 + Wu**2))

            if Cm is not None and Cu is not None and Cu != 0: set_val(f'alpha{s_i}', math.degrees(math.atan2(Cm, Cu)))
            if alpha is not None and Cm is not None:
                r = math.radians(alpha)
                if abs(math.sin(r)) > 1e-5: set_val(f'Cu{s_i}', Cm / math.tan(r))
            if alpha is not None and Cu is not None:
                r = math.radians(alpha)
                set_val(f'Cm{s_i}', Cu * math.tan(r))

            if Cm is not None and Wu is not None and Wu != 0: set_val(f'beta{s_i}', math.degrees(math.atan2(Cm, Wu)))
            if beta is not None and Cm is not None:
                r = math.radians(beta)
                if abs(math.sin(r)) > 1e-5: set_val(f'Wu{s_i}', Cm / math.tan(r))
            if beta is not None and Wu is not None:
                r = math.radians(beta)
                set_val(f'Cm{s_i}', Wu * math.tan(r))

    req = ['omega', 'U1', 'Cm1', 'Cu1', 'U2', 'Cm2', 'Cu2']
    v['is_complete'] = all(v.get(k) is not None for k in req)

    if v['is_complete']:
        u1, cu1 = v['U1'], v['Cu1']
        u2, cu2 = v['U2'], v['Cu2']
        mdot = rho * v['Q']

        if maq == 'Bomba Centrífuga':
            v['H_teo'] = (u2 * cu2 - u1 * cu1) / 9.81
            v['Torque'] = mdot * abs(u2 * cu2 - u1 * cu1) / v['omega']
            v['U_in'], v['Cm_in'], v['Cu_in'], v['Wu_in'], v['W_in'], v['C_in'], v['alpha_in_flow'], v['beta_in_flow'] = u1, v['Cm1'], cu1, v['Wu1'], v['W1'], v['C1'], v.get('alpha1'), v.get('beta1')
            v['U_out'], v['Cm_out'], v['Cu_out'], v['Wu_out'], v['W_out'], v['C_out'], v['alpha_out_flow'], v['beta_out_flow'] = u2, v['Cm2'], cu2, v['Wu2'], v['W2'], v['C2'], v.get('alpha2'), v.get('beta2')
        else:
            v['H_teo'] = (u1 * cu1 - u2 * cu2) / 9.81
            v['Torque'] = mdot * abs(u1 * cu1 - u2 * cu2) / v['omega']
            v['U_in'], v['Cm_in'], v['Cu_in'], v['Wu_in'], v['W_in'], v['C_in'], v['alpha_in_flow'], v['beta_in_flow'] = u2, v['Cm2'], cu2, v['Wu2'], v['W2'], v['C2'], v.get('alpha2'), v.get('beta2')
            v['U_out'], v['Cm_out'], v['Cu_out'], v['Wu_out'], v['W_out'], v['C_out'], v['alpha_out_flow'], v['beta_out_flow'] = u1, v['Cm1'], cu1, v['Wu1'], v['W1'], v['C1'], v.get('alpha1'), v.get('beta1')
            
        v['Potencia'] = v['Torque'] * v['omega']
        v['m_dot'] = mdot
        v['incidence_in'] = 0
        v['incidence_out'] = 0

    return v

def plot_triangulo_velocidades(U, Cm, Cu, title):
    """
    Gera gráfico interativo Plotly do triângulo de velocidades com eixos ajustáveis e ângulos.
    """
    fig = go.Figure()
    
    # Vetor C (Absoluta)
    fig.add_trace(go.Scatter(x=[0, Cu], y=[0, Cm], mode='lines+text', name='C (Absoluta)',
                             text=['', 'C'], textposition="top center", line=dict(color='red', width=4),
                             showlegend=True))
    
    # Vetor U (Pá)
    fig.add_trace(go.Scatter(x=[0, U], y=[0, 0], mode='lines+text', name='U (Pá)',
                             text=['', 'U'], textposition="bottom right", line=dict(color='blue', width=4),
                             showlegend=True))
    
    # Vetor W (Relativa)
    fig.add_trace(go.Scatter(x=[U, Cu], y=[0, Cm], mode='lines+text', name='W (Relativa)',
                             text=['', 'W'], textposition="top right", line=dict(color='green', width=4),
                             showlegend=True))
                             
    # Adicionando setas para as extremidades (Annotations)
    fig.add_annotation(x=Cu, y=Cm, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='red')
    fig.add_annotation(x=U, y=0, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='blue')
    fig.add_annotation(x=Cu, y=Cm, ax=U, ay=0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='green')

    # Cálculos dos ângulos
    alpha_deg = np.degrees(np.arctan2(Cm, Cu)) if Cu != 0 else 90.0
    beta_deg = np.degrees(np.arctan2(Cm, abs(Cu - U))) if (Cu - U) != 0 else 90.0
    
    # Posicionamento do texto dos ângulos
    text_alpha_x = Cu / 2 if Cu > 0 else -abs(Cu / 2)
    text_beta_x = U + ((Cu - U) / 2)
    
    # Textos dos ângulos
    fig.add_annotation(x=0, y=Cm*0.1, text=f"α = {abs(alpha_deg):.1f}°", showarrow=False, xanchor="left" if Cu >= 0 else "right", font=dict(size=12))
    fig.add_annotation(x=U, y=Cm*0.1, text=f"β = {abs(beta_deg):.1f}°", showarrow=False, xanchor="right" if Cu < U else "left", font=dict(size=12))

    # Limites para autoscale com padding proporcional
    min_x = min(0, U, Cu)
    max_x = max(0, U, Cu)
    pad_x = (max_x - min_x) * 0.15 if (max_x - min_x) > 0 else U * 0.15
    pad_y = Cm * 0.2 if Cm > 0 else 5
    
    fig.update_layout(
        title=title, 
        xaxis_title="Velocidade Tangencial (m/s)", 
        yaxis_title="Velocidade Meridional (m/s)",
        xaxis=dict(range=[min_x - pad_x, max_x + pad_x]),
        yaxis=dict(range=[-max(pad_y, 0), Cm + pad_y]),
        showlegend=True, 
        height=450, 
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#f4f4f5"
    )
    
    # Mantém a mesma escala x e y para ângulos reais
    fig.update_yaxes(scaleanchor="x", scaleratio=1) 
    
    return fig

# ==========================================
# INTERFACE DO USUÁRIO - SIDEBAR (INPUTS)
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Centrifugal_Pump.png/800px-Centrifugal_Pump.png", width=250)
st.sidebar.title("Setup Global")

maq_tipo = st.sidebar.radio("Tipo de Máquina:", ["Bomba Centrífuga", "Turbina Hidráulica"])

st.sidebar.markdown("---")
st.sidebar.subheader("💧 Propriedades do Fluido")
rho = st.sidebar.number_input("Densidade (kg/m³)", value=998.2, step=1.0)
mu = st.sidebar.number_input("Viscosidade (Pa.s)", value=0.001003, format="%.6f")
Z = st.sidebar.number_input("Número de Pás (Z)", value=6, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Condições Iterativas")
st.sidebar.markdown("<small>Deixe em branco (None) para resolver, ou digite o valor.</small>", unsafe_allow_html=True)

raw_inputs = {}

def get_input(label, default_val=None):
    return st.sidebar.number_input(label, value=default_val)

raw_inputs['Q'] = get_input("Vazão Q (m³/s)", 0.1)
raw_inputs['N'] = get_input("Rotação N (RPM)", 1750.0)

with st.sidebar.expander("Estação 1", expanded=True):
    for prop in ['D', 'b', 'alpha', 'beta', 'U', 'Cm', 'Cu', 'Wu', 'C', 'W']:
        key = f'{prop}1'
        default = {'D1': 0.15, 'b1': 0.04, 'alpha1': 90.0, 'beta1': 22.0}.get(key, None)
        raw_inputs[key] = st.number_input(f"{prop}1", value=default, key=f"in_{key}")

with st.sidebar.expander("Estação 2", expanded=True):
    for prop in ['D', 'b', 'alpha', 'beta', 'U', 'Cm', 'Cu', 'Wu', 'C', 'W']:
        key = f'{prop}2'
        default = {'D2': 0.30, 'b2': 0.02, 'alpha2': 20.0, 'beta2': 25.0}.get(key, None)
        raw_inputs[key] = st.number_input(f"{prop}2", value=default, key=f"in_{key}")

# ==========================================
# EXECUÇÃO DO MOTOR DE CÁLCULO
# ==========================================
try:
    calc = calcular_cinematica_iterativa(maq_tipo, rho, raw_inputs)
    calc_success = calc.get('is_complete', False)
    if not calc_success:
        st.sidebar.error("Triângulo de velocidades incompleto. Preencha mais variáveis.")
except Exception as e:
    st.error(f"Erro inesperado na cinemática: {e}")
    calc_success = False

# ==========================================
# ÁREA CENTRAL - TABS
# ==========================================
st.title("🌊 Plataforma Avançada de Turbomáquinas: Design & CFD")
st.markdown("**Engenharia Unificada:** Análise 1D Euler ➔ Setup ANSYS Fluent ➔ Malha (Y+) ➔ Diagnóstico.")

if calc_success:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 1. Projeto Analítico & Cinemática", 
        "⚙️ 2. Setup CFD (ANSYS Fluent)", 
        "🕸️ 3. Malha & Turbulência (Y+)", 
        "🩺 4. Validação & Diagnóstico",
        "💬 5. Assistente de IA"
    ])

    # ------------------------------------------
    # TAB 1: PROJETO ANALÍTICO
    # ------------------------------------------
    with tab1:
        st.header("Análise 1D baseada nas Equações de Euler")
        
        if maq_tipo == "Bomba Centrífuga" and abs(calc["incidence_in"]) > 5:
            st.warning(f"⚠️ **Atenção:** Ângulo de incidência no bordo de ataque é elevado ({calc['incidence_in']:.1f}°). Risco de choque, separação precoce de camada limite e redução de eficiência.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Carga Teórica (Euler)", f"{calc['H_teo']:.2f} m")
        col2.metric("Potência Hidráulica", f"{calc['Potencia']/1000:.2f} kW")
        col3.metric("Torque (Eixo)", f"{calc['Torque']:.2f} N.m")
        col4.metric("Velocidade Específica (Ns)", f"{(calc['N'] * np.sqrt(calc['Q'])) / (calc['H_teo']**0.75):.1f}")
        
        st.markdown("### Triângulos de Velocidade (Escoamento Ideal)")
        c1, c2 = st.columns(2)
        with c1:
            fig_in = plot_triangulo_velocidades(calc["U_in"], calc["Cm_in"], calc["Cu_in"], "Entrada do Rotor")
            st.plotly_chart(fig_in, use_container_width=True)
        with c2:
            fig_out = plot_triangulo_velocidades(calc["U_out"], calc["Cm_out"], calc["Cu_out"], "Saída do Rotor")
            st.plotly_chart(fig_out, use_container_width=True)
            
        with st.expander("Ver Matriz de Velocidades (Valores Exatos)"):
            df_vel = pd.DataFrame({
                "Estação": ["Entrada", "Saída"],
                "Tangencial da Pá (U) [m/s]": [calc["U_in"], calc["U_out"]],
                "Meridional (Cm) [m/s]": [calc["Cm_in"], calc["Cm_out"]],
                "Absoluta Tangencial (Cu) [m/s]": [calc["Cu_in"], calc["Cu_out"]],
                "Relativa Tangencial (Wu) [m/s]": [calc["Wu_in"], calc["Wu_out"]],
                "Absoluta (C) [m/s]": [calc["C_in"], calc["C_out"]],
                "Relativa (W) [m/s]": [calc["W_in"], calc["W_out"]],
                "Ângulo Absoluto (α) [°]": [calc["alpha_in_flow"], calc["alpha_out_flow"]],
                "Ângulo Relativo (β) [°]": [calc["beta_in_flow"], calc["beta_out_flow"]],
            })
            st.dataframe(df_vel.style.format(precision=2))

    # ------------------------------------------
    # TAB 2: SETUP DE CFD
    # ------------------------------------------
    with tab2:
        st.header("Parâmetros Exatos para Inserção no ANSYS Fluent")
        
        st.subheader("1. Domínio Rotacional (Cell Zone Conditions -> MRF)")
        st.info(f"👉 **Frame Motion / Mesh Motion:** Ativar na zona fluida do rotor.\n\n"
                f"**Rotational Velocity:** `{calc['omega']:.4f} rad/s` (equivalente a {calc['N']:.1f} RPM).\n\n"
                f"**Rotation-Axis Direction:** Verificar a regra da mão direita no seu CAD.")
        
        st.subheader("2. Condições de Contorno (Boundary Conditions)")
        c_bc1, c_bc2 = st.columns(2)
        
        with c_bc1:
            st.markdown("#### INLET")
            if maq_tipo == "Bomba Centrífuga":
                st.markdown("**Tipo Sugerido:** `Mass Flow Inlet` ou `Velocity Inlet`")
                st.markdown(f"- **Mass Flow Rate:** `{calc['m_dot']:.3f} kg/s`")
            else:
                st.markdown("**Tipo Sugerido:** `Pressure Inlet` ou `Mass Flow Inlet`")
                st.markdown(f"- Carga Pressão: `{(calc['H_teo'] * 9.81 * rho):.0f} Pa`")
                st.markdown(f"- Vazão Mássica: `{calc['m_dot']:.3f} kg/s`")
                
        with c_bc2:
            st.markdown("#### OUTLET")
            if maq_tipo == "Bomba Centrífuga":
                st.markdown("**Tipo Sugerido:** `Pressure Outlet`")
                st.markdown("- **Gauge Pressure:** `0 Pa`")
                st.markdown(f"- **Backflow Hydraulic Diameter:** `{2 * calc.get('b2', 0):.4f} m`")
            else:
                st.markdown("**Tipo Sugerido:** `Pressure Outlet`")

    # ------------------------------------------
    # TAB 3: MALHA & CALCULADORA Y+
    # ------------------------------------------
    with tab3:
        st.header("Dimensionamento da Primeira Camada de Malha")
        col_y1, col_y2 = st.columns([1, 2])
        
        with col_y1:
            mod_turb = st.selectbox("Modelo Alvo", ["k-omega SST (Y+ < 5)", "k-epsilon Default (Y+ > 30)"])
            y_target = st.number_input("Y+ Desejado", value=1.0 if "SST" in mod_turb else 50.0)
            beta_med = (calc.get('beta1', 0) + calc.get('beta2', 0))/2
            L_est = ((calc.get('D2', 0)/2) - (calc.get('D1', 0)/2)) / math.cos(math.radians(beta_med)) if beta_med != 90 else 0.1
            L_char = st.number_input("Corda Estimada (m)", value=float(L_est), format="%.4f")
            
        with col_y2:
            W_med = (calc.get("W_in", 0) + calc.get("W_out", 0)) / 2.0
            Re = (rho * W_med * L_char) / mu if (mu > 0 and W_med > 0) else float('nan')
            Cf = ((2 * math.log10(Re) - 0.65) ** -2.3) if (Re > 0 and not math.isnan(Re)) else float('nan')
            Tau_w = 0.5 * Cf * rho * (W_med ** 2) if not math.isnan(Cf) else float('nan')
            U_tau = np.sqrt(Tau_w / rho) if (Tau_w > 0 and not math.isnan(Tau_w)) else float('nan')
            delta_y = (y_target * mu) / (rho * U_tau) if (U_tau > 0 and not math.isnan(U_tau)) else float('nan')
            
            st.success(f"📏 **Altura da Primeira Célula (Δy):** {delta_y * 1000.0:.5f} mm")
            st.write(f"- $W_{{med}}$: {W_med:.2f} m/s | Reynolds: {Re:.2e} | Cf (Schlichting): {Cf:.5f}")

    # ------------------------------------------
    # TAB 4: VALIDAÇÃO & DIAGNÓSTICO
    # ------------------------------------------
    with tab4:
        st.header("Diagnóstico de CFD")
        with st.form("cfd_results"):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                torque_cfd = st.number_input("Torque Extraído (N.m)", value=calc.get("Torque", 0)*0.85)
            with col_r2:
                delta_p_cfd = st.number_input("Delta P Extraído (Pa)", value=calc.get("H_teo", 0)*rho*9.81*0.8)
            with col_r3:
                beta2_cfd = st.number_input("Âng. Saída β2_cfd (°)", value=calc.get('beta2', 25) - 3.0)
                
            if st.form_submit_button("Gerar Diagnóstico"):
                head_cfd = delta_p_cfd / (rho * 9.81)
                err_head = ((head_cfd - calc["H_teo"]) / calc["H_teo"]) * 100
                err_torque = ((torque_cfd - calc["Torque"]) / calc["Torque"]) * 100
                slip = calc.get('beta2', 0) - beta2_cfd
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Carga", f"{head_cfd:.2f} m", f"{err_head:.1f}%", delta_color="inverse")
                c_res2.metric("Torque", f"{torque_cfd:.2f} N.m", f"{err_torque:.1f}%", delta_color="inverse")
                c_res3.metric("Slip", f"{slip:.1f}°", "Ideal: 0°", delta_color="inverse")

    # ------------------------------------------
    # TAB 5: ASSISTENTE DE IA
    # ------------------------------------------
    with tab5:
        st.header("Assistente de IA Integrado (Especialista em CFD)")
        st.markdown("Descreva qualitativamente o que você observou nos resultados de CFD (por exemplo: 'recirculação na saída', 'zona de baixa pressão no bordo de ataque') para receber um diagnóstico especializado.")
        
        # Obter API Key
        api_key = st.text_input("Sua Chave de API do Gemini (Google AI Studio):", type="password", help="Insira sua chave para usar a IA. Ela só é usada nesta sessão.")
        
        user_issue = st.text_area("Qual o problema observado?", placeholder="Ex: Estou observando grande recirculação na saída da pá, próxima a seção externa (Shroud)...", height=120)
        
        if st.button("Gerar Diagnóstico pela IA", type="primary"):
            if not api_key.strip():
                st.error("Por favor, forneça sua Chave de API do Gemini para continuar.")
            elif not user_issue.strip():
                st.warning("Por favor, descreva o problema observado.")
            else:
                try:
                    with st.spinner("Analisando fisicamente o problema..."):
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""
Você é um Engenheiro Sênior Especialista em Turbomáquinas, CFD (Computational Fluid Dynamics) e Dinâmica dos Fluidos.

O usuário está relatando o seguinte problema observado em sua máquina ({maq_tipo}):
"{user_issue}"

Os parâmetros atuais de projeto e de cinemática (Teoria de Euler) da máquina são:
- Rotação: {calc.get('N', 0):.1f} RPM
- Vazão: {calc.get('Q', 0):.3f} m³/s
- Diâmetros: D1={calc.get('D1', 0):.3f}m, D2={calc.get('D2', 0):.3f}m
- Ângulos das Pás: Beta1={calc.get('beta1', 0):.1f}°, Beta2={calc.get('beta2', 0):.1f}°
- Resultados Cinemáticos Ideais: Carga Teórica={calc['H_teo']:.2f}m, Potência={calc['Potencia']/1000:.2f}kW, Torque={calc['Torque']:.2f}Nm.
- Velocidades (Entrada): U={calc['U_in']:.2f}m/s, Cm={calc['Cm_in']:.2f}m/s, W={calc['W_in']:.2f}m/s
- Velocidades (Saída): U={calc['U_out']:.2f}m/s, Cm={calc['Cm_out']:.2f}m/s, W={calc['W_out']:.2f}m/s
- Incidência Teórica no Bordo de Ataque: {calc['incidence_in']:.2f}°

Baseado no relato do usuário e nos parâmetros matemáticos, faça um diagnóstico focado. 
Sua resposta deve conter:
1. Uma **Análise do Problema Físico** (por que esse sintoma ocorre em turbomáquinas com base na geometria ou cinemática atual).
2. **Impacto no Desempenho** (o que esperar na perda de rendimento, carga ou riscos estruturais).
3. **Solução Recomendada** (Sugestões de alterações de projeto: alterar ângulos, aumentar/diminuir corda, alterar Z, etc. ou sugestão de setup de CFD).

Responda em Português, de forma técnica e objetiva em Markdown.
"""
                        response = model.generate_content(prompt)
                        st.subheader("Diagnóstico da IA")
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Erro na comunicação com a API: {e}")
