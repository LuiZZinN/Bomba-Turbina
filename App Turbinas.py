import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hydro Sizer Pro", layout="wide", page_icon="🌊")

st.title("🌊 Dimensionamento de Turbinas Hidráulicas")
st.markdown("""
Este aplicativo interativo realiza o pré-dimensionamento, classificação estatística e a modelagem 
cinemática (1D/Euler) de turbinas hidráulicas para setups de CFD.
""")

def solve_hydro(Q, H, f, p, eta, advanced, cRPM, D1, B1, a1, D2, a2):
    rho = 1000
    g = 9.81
    if p <= 0 or H <= 0 or Q <= 0: return None
    
    sync_N = (60 * f) / p
    N = cRPM if (advanced and cRPM > 0) else sync_N
    
    P_W = eta * rho * g * Q * H
    P_kW = P_W / 1000
    P_MW = P_kW / 1000
    
    Ns = (N * math.sqrt(P_kW)) / (H ** 1.25)
    if Ns < 35: t_type = 'Pelton'
    elif Ns < 350: t_type = 'Francis'
    else: t_type = 'Kaplan'
    
    Ku = 0.31 + (0.001 * Ns)
    sqrt2gH = math.sqrt(2 * g * H)
    U1_emp = Ku * sqrt2gH
    D1_emp = (60 * U1_emp) / (math.pi * N)
    D2_emp = D1_emp * 0.6
    Cm1_emp = 0.2 * sqrt2gH
    B1_emp = Q / (math.pi * D1_emp * Cm1_emp)
    
    D1_real = D1 if (advanced and D1 > 0) else D1_emp
    B1_real = B1 if (advanced and B1 > 0) else B1_emp
    a1_real = a1 if advanced else 15
    D2_real = D2 if (advanced and D2 > 0) else D2_emp
    a2_real = a2 if advanced else 90
    
    omega = (N * math.pi) / 30
    massFlow = Q * rho
    
    def to_rad(d): return d * math.pi / 180
    def to_deg(r): return r * 180 / math.pi
    
    U1 = (math.pi * D1_real * N) / 60
    Cm1 = Q / (math.pi * D1_real * B1_real)
    Cu1 = 0 if a1_real == 90 else Cm1 / math.tan(to_rad(a1_real))
    C1 = math.sqrt(Cm1**2 + Cu1**2)
    Wu1 = U1 - Cu1
    W1 = math.sqrt(Cm1**2 + Wu1**2)
    beta1 = to_deg(math.atan2(Cm1, Wu1))
    inlet = {"U": U1, "Cm": Cm1, "Cu": Cu1, "C": C1, "Wu": Wu1, "W": W1, "alpha": a1_real, "beta": beta1}
    
    U2 = (math.pi * D2_real * N) / 60
    Area2 = (math.pi * D2_real**2) / 4
    Cm2 = Q / Area2
    Cu2 = 0 if a2_real == 90 else Cm2 / math.tan(to_rad(a2_real))
    C2 = math.sqrt(Cm2**2 + Cu2**2)
    Wu2 = U2 - Cu2
    W2 = math.sqrt(Cm2**2 + Wu2**2)
    beta2 = to_deg(math.atan2(Cm2, Wu2))
    outlet = {"U": U2, "Cm": Cm2, "Cu": Cu2, "C": C2, "Wu": Wu2, "W": W2, "alpha": a2_real, "beta": beta2}
    
    EulerWork = (U1 * Cu1) - (U2 * Cu2)
    Heuler = EulerWork / g
    eulerPowerW = massFlow * EulerWork
    effHyd = Heuler / H if H > 0 else 0
    eulerTorque = eulerPowerW / omega if omega > 0 else 0
    globalTorque = P_W / omega if omega > 0 else 0
    
    return {
        "N": N, "P_MW": P_MW, "P_kW": P_kW, "Ns": Ns, "type": t_type, "D1": D1_real, "D2": D2_real,
        "omega": omega, "massFlow": massFlow, "Heuler": Heuler, "Pe_kW": eulerPowerW/1000,
        "effHyd": effHyd, "Te": eulerTorque, "Tg": globalTorque, "inlet": inlet, "outlet": outlet,
        "H": H
    }

with st.sidebar:
    st.header("⚙️ Entradas Principais")
    Q = st.number_input("Vazão (Q) [m³/s]", value=15.0, min_value=0.1)
    H = st.number_input("Queda Líquida (H) [m]", value=120.0, min_value=0.1)
    f = st.selectbox("Frequência [Hz]", [50, 60], index=1)
    p = st.number_input("Pares Polo (p)", value=6, min_value=1)
    eta = st.number_input("Eficiência Global Est. (η)", value=0.88, max_value=1.0)
    
    st.markdown("---")
    advanced = st.checkbox("Modo Cinemático & Geometria")
    cRPM, D1, B1, a1, D2, a2 = 0,0,0,0,0,0
    if advanced:
        cRPM = st.number_input("RPM Forçado [RPM]", value=600.0)
        D1 = st.number_input("D1 (Entrada) [m]", value=1.5)
        B1 = st.number_input("B1 (Altura) [m]", value=0.3)
        a1 = st.number_input("α₁ (Ataque) [°]", value=15.0)
        D2 = st.number_input("D2 (Saída) [m]", value=0.9)
        a2 = st.number_input("α₂ (Saída) [°]", value=90.0, help="90° = sem redemoinho")

res = solve_hydro(Q, H, f, p, eta, advanced, cRPM, D1, B1, a1, D2, a2)

if not res:
    st.error("Valores de entrada inválidos. Verifique os dados inseridos.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["1. Resultados Globais", "2. Cinemática (Euler)", "3. Triângulos de Velocidades", "4. CFD & Report"])
    
    with tab1:
        st.subheader(f"Turbina Indicada: **{res['type']}**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rot. Específica (Ns)", f"{res['Ns']:.1f}")
        col2.metric("Potência Base (Global)", f"{res['P_MW']:.2f} MW")
        col3.metric("Rotação Síncrona", f"{res['N']:.1f} RPM")
        col4.metric("Diâmetro D1 (Est.)", f"{res['D1']:.2f} m")
        
    with tab2:
        if not advanced:
            st.info("Ative o **Modo Cinemático** na barra lateral para definir parâmetros analíticos de Euler.")
        
        st.subheader("Balanço de Quantidade de Movimento")
        c1, c2, c3 = st.columns(3)
        c1.metric("Carga Específica (Heuler)", f"{res['Heuler']:.1f} m", help="Energia bruta extraída")
        c2.metric("Potência (Euler)", f"{res['Pe_kW']:.0f} kW")
        c3.metric("Rend. Hidráulico (η_h)", f"{res['effHyd']*100:.1f} %")
        
    with tab3:
        st.subheader("Vetores Cinemáticos")
        if not advanced:
            st.warning("O modo cinemático está desativado. Os triângulos abaixo representam valores empíricos genéricos.")
            
        def plot_triangle(tri, title):
            fig, ax = st.subplots(figsize=(6, 4))
            # Vetor U
            ax.annotate(f"U={tri['U']:.1f}", xy=(tri['U'], 0), xytext=(tri['U']/2, -tri['Cm']*0.1),
                        arrowprops=dict(color='blue', width=2, headwidth=8), color='blue')
            # Vetor C
            ax.annotate(f"C={tri['C']:.1f}", xy=(tri['Cu'], tri['Cm']), xytext=(tri['Cu']/2, tri['Cm']/2),
                        arrowprops=dict(color='red', width=2, headwidth=8), color='red')
            # Vetor W
            ax.annotate(f"W={tri['W']:.1f}", xy=(tri['Cu'], tri['Cm']), xytext=(tri['U'] - tri['Wu']/2, tri['Cm']/2),
                        arrowprops=dict(color='green', width=2, headwidth=8), color='green')
            
            ax.set_xlim(min(0, tri['Cu'], tri['U']) - max(tri['U'], tri['Cm'])*0.1, max(tri['U'], tri['Cu']) * 1.2)
            ax.set_ylim(-max(tri['U'], tri['Cm'])*0.1, tri['Cm'] * 1.2)
            ax.set_title(title)
            ax.axis('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            return fig
            
        col_t1, col_t2 = st.columns(2)
        with col_t1: st.pyplot(plot_triangle(res['inlet'], "Entrada no Rotor (Seção 1)"))
        with col_t2: st.pyplot(plot_triangle(res['outlet'], "Saída do Rotor (Seção 2)"))
        
    with tab4:
        st.subheader("Setup para ANSYS / CFX")
        script = f'''// CONDIÇÕES DE OPERAÇÃO
Rotational_Speed_rads = {res['omega']:.4f} [rad/s]
Rotational_Speed_RPM  = {res['N']:.2f} [rpm]
Mass_Flow_Rate        = {res['massFlow']:.2f} [kg/s]

// GEOMETRIA E ANGULOS
Diameter_Outer_D1     = {res['D1']:.3f} [m]
Diameter_Inner_D2     = {res['D2']:.3f} [m]
Angle_Beta_1          = {res['inlet']['beta']:.2f} [deg]
Angle_Beta_2          = {res['outlet']['beta']:.2f} [deg]

// ALVOS TEÓRICOS DE TORQUE E POTÊNCIA
Target_Torque_Global  = {res['Tg']:.2f} [N.m]
Target_Torque_Euler   = {res['Te']:.2f} [N.m]
'''
        st.code(script, language='c')
        
        st.markdown("### Validação Pós-CFD")
        simTorque = st.number_input("Torque Medido no Eixo CFD [N·m]", value=float(f"{res['Te']:.2f}"))
        simHead = st.number_input("Queda Total Efetiva CFD [m] (Opcional)", value=float(f"{res['Heuler']:.2f}"))
        
        if simTorque > 0:
            pSimW = simTorque * res['omega']
            pSimkW = pSimW / 1000
            st.success(f"**Potência Atingida CFD:** {pSimkW:.1f} kW")
            
            colA, colB = st.columns(2)
            colA.metric("Desvio vs Meta Global (1D)", f"{(simTorque / res['Tg']) * 100:.1f} %")
            colB.metric("Desvio vs Meta Euler", f"{(simTorque / res['Te']) * 100:.1f} %")
            
            if simHead > 0:
                g = 9.81
                hydPowerCFD = res['massFlow'] * g * simHead
                effReal = (pSimW / hydPowerCFD) * 100
                st.info(f"**Eficiência Fluido-Mecânica (CFD):** {effReal:.1f} %")