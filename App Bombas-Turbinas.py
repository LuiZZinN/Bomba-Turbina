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
def calcular_cinematica(maq, rho, Q, N, D1, D2, b1, b2, beta1, beta2):
    """
    Motor de cálculo analítico e cinemático.
    Lida automaticamente com a inversão da física entre Bomba e Turbina.
    """
    g = 9.81
    omega = 2 * np.pi * N / 60.0
    
    # Setup de entrada e saída baseado na máquina
    if maq == "Bomba Centrífuga":
        D_in, D_out = D1, D2
        b_in, b_out = b1, b2
        beta_in, beta_out = np.radians(beta1), np.radians(beta2)
    else: # Turbina Hidráulica
        D_in, D_out = D2, D1
        b_in, b_out = b2, b1
        beta_in, beta_out = np.radians(beta2), np.radians(beta1)

    # Velocidades Tangenciais do Rotor (U)
    U_in = omega * (D_in / 2.0)
    U_out = omega * (D_out / 2.0)

    # Velocidades Meridionais (Cm) - Ignorando espessura da pá
    Cm_in = Q / (np.pi * D_in * b_in)
    Cm_out = Q / (np.pi * D_out * b_out)

    # Triângulo de Entrada
    Wu_in = Cm_in / np.tan(beta_in)
    Cu_in = U_in - Wu_in
    W_in = np.sqrt(Cm_in**2 + Wu_in**2)
    C_in = np.sqrt(Cm_in**2 + Cu_in**2)
    
    flow_angle_in = np.degrees(np.arctan(Cm_in / U_in)) if maq == "Bomba Centrífuga" else np.degrees(beta_in)
    incidence = np.degrees(beta_in) - flow_angle_in

    # Triângulo de Saída
    Wu_out = Cm_out / np.tan(beta_out)
    Cu_out = U_out - Wu_out
    W_out = np.sqrt(Cm_out**2 + Wu_out**2)
    C_out = np.sqrt(Cm_out**2 + Cu_out**2)

    # Equações de Euler
    if maq == "Bomba Centrífuga":
        H_teo = (U_out * Cu_out - U_in * Cu_in) / g
    else:
        H_teo = (U_in * Cu_in - U_out * Cu_out) / g

    m_dot = rho * Q
    if maq == "Bomba Centrífuga":
        Torque = m_dot * abs(U_out * Cu_out/omega - U_in * Cu_in/omega)
    else:
        Torque = m_dot * abs(U_in * Cu_in/omega - U_out * Cu_out/omega)
        
    Potencia = Torque * omega

    return {
        "omega": omega, "m_dot": m_dot, "H_teo": H_teo, "Torque": Torque, "Potencia": Potencia,
        "U_in": U_in, "Cm_in": Cm_in, "Cu_in": Cu_in, "Wu_in": Wu_in, "W_in": W_in, "C_in": C_in,
        "U_out": U_out, "Cm_out": Cm_out, "Cu_out": Cu_out, "Wu_out": Wu_out, "W_out": W_out, "C_out": C_out,
        "incidence": incidence
    }

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

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Condições Operacionais")
Q = st.sidebar.number_input("Vazão Q (m³/s)", value=0.1, step=0.01)
N = st.sidebar.number_input("Rotação N (RPM)", value=1750.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Geometria do Rotor")
D1 = st.sidebar.number_input("Diâmetro D1 - Interno (m)", value=0.15, step=0.01)
D2 = st.sidebar.number_input("Diâmetro D2 - Externo (m)", value=0.30, step=0.01)
b1 = st.sidebar.number_input("Largura b1 (m)", value=0.04, step=0.005)
b2 = st.sidebar.number_input("Largura b2 (m)", value=0.02, step=0.005)
beta1 = st.sidebar.number_input("Ângulo Pá Entrada β1 (graus)", value=22.0, step=1.0)
beta2 = st.sidebar.number_input("Ângulo Pá Saída β2 (graus)", value=25.0, step=1.0)
Z = st.sidebar.number_input("Número de Pás (Z)", value=6, step=1)

# ==========================================
# EXECUÇÃO DO MOTOR DE CÁLCULO
# ==========================================
try:
    calc = calcular_cinematica(maq_tipo, rho, Q, N, D1, D2, b1, b2, beta1, beta2)
    calc_success = True
except ZeroDivisionError:
    st.error("Erro: Parâmetro geométrico não pode ser zero.")
    calc_success = False
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
        
        if maq_tipo == "Bomba Centrífuga" and abs(calc["incidence"]) > 5:
            st.warning(f"⚠️ **Atenção:** Ângulo de incidência no bordo de ataque é elevado ({calc['incidence']:.1f}°). Risco de choque, separação precoce de camada limite e redução de eficiência.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Carga Teórica (Euler)", f"{calc['H_teo']:.2f} m")
        col2.metric("Potência Hidráulica", f"{calc['Potencia']/1000:.2f} kW")
        col3.metric("Torque (Eixo)", f"{calc['Torque']:.2f} N.m")
        col4.metric("Velocidade Específica (Ns)", f"{(N * np.sqrt(Q)) / (calc['H_teo']**0.75):.1f}")
        
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
                "Relativa (W) [m/s]": [calc["W_in"], calc["W_out"]]
            })
            st.dataframe(df_vel.style.format(precision=2))

    # ------------------------------------------
    # TAB 2: SETUP DE CFD
    # ------------------------------------------
    with tab2:
        st.header("Parâmetros Exatos para Inserção no ANSYS Fluent")
        
        st.subheader("1. Domínio Rotacional (Cell Zone Conditions -> MRF)")
        st.info(f"👉 **Frame Motion / Mesh Motion:** Ativar na zona fluida do rotor.\n\n"
                f"**Rotational Velocity:** `{calc['omega']:.4f} rad/s` (equivalente a {N} RPM).\n\n"
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
                st.markdown(f"- **Backflow Hydraulic Diameter:** `{2 * b2:.4f} m`")
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
            beta_med = (beta1 + beta2)/2
            L_est = ((D2/2) - (D1/2)) / math.cos(math.radians(beta_med))
            L_char = st.number_input("Corda Estimada (m)", value=float(L_est), format="%.4f")
            
        with col_y2:
            W_med = (calc["W_in"] + calc["W_out"]) / 2.0
            Re = (rho * W_med * L_char) / mu
            Cf = 0.0576 * (Re ** -0.2)
            Tau_w = 0.5 * Cf * rho * (W_med ** 2)
            U_tau = np.sqrt(Tau_w / rho)
            delta_y = (y_target * mu) / (rho * U_tau)
            
            st.success(f"📏 **Altura da Primeira Célula (Δy):** {delta_y * 1000.0:.5f} mm")
            st.write(f"- $W_{{med}}$: {W_med:.2f} m/s | Reynolds: {Re:.2e}")

    # ------------------------------------------
    # TAB 4: VALIDAÇÃO & DIAGNÓSTICO
    # ------------------------------------------
    with tab4:
        st.header("Diagnóstico de CFD")
        with st.form("cfd_results"):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                torque_cfd = st.number_input("Torque Extraído (N.m)", value=calc["Torque"]*0.85)
            with col_r2:
                delta_p_cfd = st.number_input("Delta P Extraído (Pa)", value=calc["H_teo"]*rho*9.81*0.8)
            with col_r3:
                beta2_cfd = st.number_input("Âng. Saída β2_cfd (°)", value=beta2 - 3.0)
                
            if st.form_submit_button("Gerar Diagnóstico"):
                head_cfd = delta_p_cfd / (rho * 9.81)
                err_head = ((head_cfd - calc["H_teo"]) / calc["H_teo"]) * 100
                err_torque = ((torque_cfd - calc["Torque"]) / calc["Torque"]) * 100
                slip = beta2 - beta2_cfd
                
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
- Rotação: {N} RPM
- Vazão: {Q} m³/s
- Diâmetros: D1={D1}m, D2={D2}m
- Ângulos das Pás: Beta1={beta1}°, Beta2={beta2}°
- Resultados Cinemáticos Ideais: Carga Teórica={calc['H_teo']:.2f}m, Potência={calc['Potencia']/1000:.2f}kW, Torque={calc['Torque']:.2f}Nm.
- Velocidades (Entrada): U={calc['U_in']:.2f}m/s, Cm={calc['Cm_in']:.2f}m/s, W={calc['W_in']:.2f}m/s
- Velocidades (Saída): U={calc['U_out']:.2f}m/s, Cm={calc['Cm_out']:.2f}m/s, W={calc['W_out']:.2f}m/s
- Incidência Teórica: {calc['incidence']:.2f}°

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
