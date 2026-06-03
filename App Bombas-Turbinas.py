import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Turbomáquinas & CFD Setup", layout="wide")

# ==========================================
# 1. Kinematic Solver
# ==========================================
def solve_kinematic_system(maq, rho, inputs):
    v = inputs.copy()
    changed = True
    iters = 0

    def set_val(k, val):
        nonlocal changed
        if k not in v or v[k] is None or math.isnan(v[k]):
            v[k] = val
            changed = True

    while changed and iters < 100:
        changed = False
        iters += 1

        if v.get('N') is not None: set_val('omega', v['N'] * math.pi / 30)
        if v.get('omega') is not None: set_val('N', v['omega'] * 30 / math.pi)

        for i in [1, 2]:
            s_i = str(i)
            D = v.get('D' + s_i)
            b = v.get('b' + s_i)
            U = v.get('U' + s_i)
            Cm = v.get('Cm' + s_i)
            Cu = v.get('Cu' + s_i)
            Wu = v.get('Wu' + s_i)
            C = v.get('C' + s_i)
            W = v.get('W' + s_i)
            alpha = v.get('alpha' + s_i)
            beta = v.get('beta' + s_i)
            Q = v.get('Q')
            omega = v.get('omega')

            if U is not None and omega is not None and omega != 0: set_val('D'+s_i, 2 * U / omega)
            if U is not None and D is not None and D != 0: set_val('omega', 2 * U / D)
            if omega is not None and D is not None: set_val('U'+s_i, omega * D / 2)

            if Q is not None and D is not None and b is not None and D * b != 0: set_val('Cm'+s_i, Q / (math.pi * D * b))
            if Cm is not None and D is not None and b is not None and Cm != 0: set_val('Q', Cm * math.pi * D * b)
            if Q is not None and Cm is not None and b is not None and Cm * b != 0: set_val('D'+s_i, Q / (math.pi * Cm * b))
            if Q is not None and Cm is not None and D is not None and Cm * D != 0: set_val('b'+s_i, Q / (math.pi * Cm * D))

            if U is not None and Cu is not None: set_val('Wu'+s_i, U - Cu)
            if U is not None and Wu is not None: set_val('Cu'+s_i, U - Wu)
            if Cu is not None and Wu is not None: set_val('U'+s_i, Cu + Wu)

            if C is not None and Cm is not None: set_val('Cu'+s_i, math.sqrt(max(0, C*C - Cm*Cm)))
            if C is not None and Cu is not None: set_val('Cm'+s_i, math.sqrt(max(0, C*C - Cu*Cu)))
            if Cm is not None and Cu is not None: set_val('C'+s_i, math.sqrt(Cm*Cm + Cu*Cu))

            if W is not None and Cm is not None: set_val('Wu'+s_i, math.sqrt(max(0, W*W - Cm*Cm)))
            if W is not None and Wu is not None: set_val('Cm'+s_i, math.sqrt(max(0, W*W - Wu*Wu)))
            if Cm is not None and Wu is not None: set_val('W'+s_i, math.sqrt(Cm*Cm + Wu*Wu))

            if Cm is not None and Cu is not None and Cu != 0: set_val('alpha'+s_i, math.degrees(math.atan2(Cm, Cu)))
            if alpha is not None and Cm is not None:
                r = math.radians(alpha)
                if abs(math.sin(r)) > 1e-5: set_val('Cu'+s_i, Cm / math.tan(r))
            if alpha is not None and Cu is not None:
                r = math.radians(alpha)
                set_val('Cm'+s_i, Cu * math.tan(r))
            if alpha is not None and C is not None:
                r = math.radians(alpha)
                set_val('Cu'+s_i, C * math.cos(r))
                set_val('Cm'+s_i, C * math.sin(r))

            if Cm is not None and Wu is not None and Wu != 0: set_val('beta'+s_i, math.degrees(math.atan2(Cm, Wu)))
            if beta is not None and Cm is not None:
                r = math.radians(beta)
                if abs(math.sin(r)) > 1e-5: set_val('Wu'+s_i, Cm / math.tan(r))
            if beta is not None and Wu is not None:
                r = math.radians(beta)
                set_val('Cm'+s_i, Wu * math.tan(r))
            if beta is not None and W is not None:
                r = math.radians(beta)
                set_val('Wu'+s_i, W * math.cos(r))
                set_val('Cm'+s_i, W * math.sin(r))

        u1_cu1 = v['U1'] * v['Cu1'] if v.get('U1') is not None and v.get('Cu1') is not None else (0 if v.get('Cu1')==0 else None)
        u2_cu2 = v['U2'] * v['Cu2'] if v.get('U2') is not None and v.get('Cu2') is not None else (0 if v.get('Cu2')==0 else None)

        if maq == 'Bomba Centrífuga':
            if v.get('H_teo') is not None and u1_cu1 is not None:
                target = v['H_teo'] * 9.81 + u1_cu1
                if v.get('U2') is not None and v['U2'] != 0: set_val('Cu2', target / v['U2'])
                if v.get('Cu2') is not None and v['Cu2'] != 0: set_val('U2', target / v['Cu2'])
            if v.get('H_teo') is not None and u2_cu2 is not None:
                target = u2_cu2 - v['H_teo'] * 9.81
                if v.get('U1') is not None and v['U1'] != 0: set_val('Cu1', target / v['U1'])
                if v.get('Cu1') is not None and v['Cu1'] != 0: set_val('U1', target / v['Cu1'])
            if u2_cu2 is not None and u1_cu1 is not None:
                set_val('H_teo', (u2_cu2 - u1_cu1) / 9.81)
        else:
            if v.get('H_teo') is not None and u2_cu2 is not None:
                target = v['H_teo'] * 9.81 + u2_cu2
                if v.get('U1') is not None and v['U1'] != 0: set_val('Cu1', target / v['U1'])
                if v.get('Cu1') is not None and v['Cu1'] != 0: set_val('U1', target / v['Cu1'])
            if v.get('H_teo') is not None and u1_cu1 is not None:
                target = u1_cu1 - v['H_teo'] * 9.81
                if v.get('U2') is not None and v['U2'] != 0: set_val('Cu2', target / v['U2'])
                if v.get('Cu2') is not None and v['Cu2'] != 0: set_val('U2', target / v['Cu2'])
            if u2_cu2 is not None and u1_cu1 is not None:
                set_val('H_teo', (u1_cu1 - u2_cu2) / 9.81)

        if v.get('H_teo') is not None and v.get('Q') is not None:
            set_val('Potencia', rho * v['Q'] * 9.81 * v['H_teo'])
        if v.get('Potencia') is not None and v.get('Q') is not None and v['Q'] != 0:
            set_val('H_teo', v['Potencia'] / (rho * v['Q'] * 9.81))
        if v.get('Potencia') is not None and v.get('H_teo') is not None and v['H_teo'] != 0:
            set_val('Q', v['Potencia'] / (rho * 9.81 * v['H_teo']))

    req = ['omega', 'U1', 'Cm1', 'Cu1', 'U2', 'Cm2', 'Cu2']
    v['is_complete'] = all(v.get(k) is not None for k in req)
    
    if v['is_complete']:
        mdot = rho * v.get('Q', 0)
        
        # Guardar velocidades finais (evitando KeyError)
        v['U_in'] = v['U1'] if maq == 'Bomba Centrífuga' else v['U2']
        v['Cm_in'] = v['Cm1'] if maq == 'Bomba Centrífuga' else v['Cm2']
        v['Cu_in'] = v['Cu1'] if maq == 'Bomba Centrífuga' else v['Cu2']
        v['Wu_in'] = v['Wu1'] if maq == 'Bomba Centrífuga' else v['Wu2']
        v['C_in'] = v['C1'] if maq == 'Bomba Centrífuga' else v['C2']
        v['W_in'] = v['W1'] if maq == 'Bomba Centrífuga' else v['W2']
        v['alpha_in'] = v['alpha1'] if maq == 'Bomba Centrífuga' else v['alpha2']
        v['beta_in'] = v['beta1'] if maq == 'Bomba Centrífuga' else v['beta2']
        
        v['U_out'] = v['U2'] if maq == 'Bomba Centrífuga' else v['U1']
        v['Cm_out'] = v['Cm2'] if maq == 'Bomba Centrífuga' else v['Cm1']
        v['Cu_out'] = v['Cu2'] if maq == 'Bomba Centrífuga' else v['Cu1']
        v['Wu_out'] = v['Wu2'] if maq == 'Bomba Centrífuga' else v['Wu1']
        v['C_out'] = v['C2'] if maq == 'Bomba Centrífuga' else v['C1']
        v['W_out'] = v['W2'] if maq == 'Bomba Centrífuga' else v['W1']
        v['alpha_out'] = v['alpha2'] if maq == 'Bomba Centrífuga' else v['alpha1']
        v['beta_out'] = v['beta2'] if maq == 'Bomba Centrífuga' else v['beta1']

        if maq == 'Bomba Centrífuga':
            v['H_teo'] = (v['U2']*v['Cu2'] - v['U1']*v['Cu1']) / 9.81
            v['Torque'] = mdot * abs(v['U2']*v['Cu2'] - v['U1']*v['Cu1']) / v['omega']
        else:
            v['H_teo'] = (v['U1']*v['Cu1'] - v['U2']*v['Cu2']) / 9.81
            v['Torque'] = mdot * abs(v['U1']*v['Cu1'] - v['U2']*v['Cu2']) / v['omega']
            
        v['Potencia'] = v['Torque'] * v['omega']
        v['m_dot'] = mdot

    return v

# ==========================================
# Application State & Title
# ==========================================
st.title("Turbomáquinas Pro - Plataforma Analítica: Design & CFD")

# ==========================================
# Sidebar: Setup Base
# ==========================================
with st.sidebar:
    st.header("Setup Global")
    maq_type = st.selectbox("Tipo de Máquina", ["Bomba Centrífuga", "Turbina Hidráulica"])
    
    col1, col2 = st.columns(2)
    rho = col1.number_input("Density ρ (kg/m³)", value=998.2, format="%.2f")
    mu = col2.number_input("Viscosity μ (Pa.s)", value=0.001003, format="%.6f")
    Z = st.number_input("Número de Pás (Z)", value=6)
    
    st.markdown("---")
    st.header("Condições Iterativas")
    st.caption("Deixe em branco para que o programa resolva o triângulo de velocidades.")
    
    # Session state for inputs to allow reset/update
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {
            'Q': 0.1, 'N': 1750.0, 'H_teo': None, 'Potencia': None,
            'D1': 0.15, 'D2': 0.3, 'b1': 0.04, 'b2': 0.02,
            'beta1': 22.0, 'beta2': 25.0, 'alpha1': 90.0, 'alpha2': 20.0,
            'U1': None, 'Cm1': None, 'Cu1': None, 'Wu1': None, 'C1': None, 'W1': None,
            'U2': None, 'Cm2': None, 'Cu2': None, 'Wu2': None, 'C2': None, 'W2': None,
        }
    
    def input_field(key, label):
        val = st.text_input(label, value=str(st.session_state.inputs.get(key, "")) if st.session_state.inputs.get(key) is not None else "", key=f"in_{key}")
        try:
            return float(val) if val.strip() != "" else None
        except ValueError:
            return None

    c1, c2 = st.columns(2)
    with c1:
        Q = input_field('Q', 'Vazão Q (m³/s)')
        H_teo = input_field('H_teo', 'Carga (m)')
    with c2:
        N = input_field('N', 'Rotação N (RPM)')
        Potencia = input_field('Potencia', 'Potência (W)')

    with st.expander("Estação 1", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            D1 = input_field('D1', 'D1')
            alpha1 = input_field('alpha1', 'alpha1')
            U1 = input_field('U1', 'U1')
            Cu1 = input_field('Cu1', 'Cu1')
            C1 = input_field('C1', 'C1')
        with c2:
            b1 = input_field('b1', 'b1')
            beta1 = input_field('beta1', 'beta1')
            Cm1 = input_field('Cm1', 'Cm1')
            Wu1 = input_field('Wu1', 'Wu1')
            W1 = input_field('W1', 'W1')

    with st.expander("Estação 2", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            D2 = input_field('D2', 'D2')
            alpha2 = input_field('alpha2', 'alpha2')
            U2 = input_field('U2', 'U2')
            Cu2 = input_field('Cu2', 'Cu2')
            C2 = input_field('C2', 'C2')
        with c2:
            b2 = input_field('b2', 'b2')
            beta2 = input_field('beta2', 'beta2')
            Cm2 = input_field('Cm2', 'Cm2')
            Wu2 = input_field('Wu2', 'Wu2')
            W2 = input_field('W2', 'W2')

    # Update session state with current inputs
    current_inputs = {
        'Q': Q, 'N': N, 'H_teo': H_teo, 'Potencia': Potencia,
        'D1': D1, 'b1': b1, 'alpha1': alpha1, 'beta1': beta1, 'U1': U1, 'Cm1': Cm1, 'Cu1': Cu1, 'Wu1': Wu1, 'C1': C1, 'W1': W1,
        'D2': D2, 'b2': b2, 'alpha2': alpha2, 'beta2': beta2, 'U2': U2, 'Cm2': Cm2, 'Cu2': Cu2, 'Wu2': Wu2, 'C2': C2, 'W2': W2,
    }
    
    # Compute system
    res = solve_kinematic_system(maq_type, rho, current_inputs)


# ==========================================
# Tabs definition
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Cinemática & Triângulos", 
    "Setup CFD (Manual)",
    "Malha & Turbulência (Y+)",
    "Validação & Diagnóstico",
    "Scripts TUI (Fluent)",
    "Assistente de IA"
])

# ==========================================
# TAB 1: Cinemática, Matriz & Triângulos
# ==========================================
with tab1:
    if not res.get('is_complete'):
        st.error("O triângulo de velocidades não está totalmente definido. Preencha mais variáveis na barra lateral.")
    else:
        st.subheader("Projeto Analítico & Cinemática")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Carga Teórica (Euler)", f"{res['H_teo']:.2f} m")
        col2.metric("Potência Hidráulica", f"{(res['Potencia'] / 1000):.2f} kW")
        col3.metric("Torque (Eixo)", f"{res['Torque']:.2f} N.m")
        ns = (res.get('N',0) * math.sqrt(res.get('Q',0))) / math.pow(res.get('H_teo',1), 0.75) if res.get('H_teo',0) > 0 else 0
        col4.metric("Velocidade Específica (Ns)", f"{ns:.1f}")

        st.markdown("### Matriz de Velocidades Exactas")
        matrix_data = {
            "Estação": ["Entrada", "Saída"],
            "Pá (U) [m/s]": [res['U_in'], res['U_out']],
            "Meridional (Cm) [m/s]": [res['Cm_in'], res['Cm_out']],
            "Abs. Tangencial (Cu) [m/s]": [res['Cu_in'], res['Cu_out']],
            "Rel. Tangencial (Wu) [m/s]": [res['Wu_in'], res['Wu_out']],
            "Absoluta (C) [m/s]": [res['C_in'], res['C_out']],
            "Relativa (W) [m/s]": [res['W_in'], res['W_out']],
            "Âng. Abs. (α) [°]": [res['alpha_in'], res['alpha_out']],
            "Âng. Rel. (β) [°]": [res['beta_in'], res['beta_out']],
        }
        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(df_matrix.style.format({col: "{:.2f}" for col in df_matrix.columns if col != "Estação"}), use_container_width=True)

        st.markdown("---")
        st.subheader("Triângulos de Velocidade (Ideal)")
        
        def create_triangle_plot(U, Cu, Cm, W_u, title):
            fig = go.Figure()
            # Vector U
            fig.add_trace(go.Scatter(x=[0, U], y=[0, 0], mode='lines+text', name='U (Periférica)',
                                     line=dict(color='green', width=3, dash='dash')))
            # Vector C
            fig.add_trace(go.Scatter(x=[0, Cu], y=[0, Cm], mode='lines+text', name='C (Absoluta)',
                                     line=dict(color='blue', width=3)))
            # Vector W (from tip of U to tip of C)
            fig.add_trace(go.Scatter(x=[U, Cu], y=[0, Cm], mode='lines+text', name='W (Relativa)',
                                     line=dict(color='red', width=3)))
            fig.update_layout(title=title, xaxis_title="Tangencial [m/s]", yaxis_title="Meridional [m/s]", 
                              yaxis=dict(scaleanchor="x", scaleratio=1), 
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig

        c1, c2 = st.columns(2)
        with c1:
            fig1 = create_triangle_plot(res['U_in'], res['Cu_in'], res['Cm_in'], res['Wu_in'], "Entrada do Rotor")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = create_triangle_plot(res['U_out'], res['Cu_out'], res['Cm_out'], res['Wu_out'], "Saída do Rotor")
            st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 2: Setup CFD (Manual)
# ==========================================
with tab2:
    if res.get('is_complete'):
        st.subheader("1. Domínio Rotacional (Cell Zone Conditions -> MRF)")
        st.markdown(f"**Rotational Velocity:** `{res['omega']:.4f} rad/s` (equivalente a {res['N']:.1f} RPM)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### INLET")
            if maq_type == 'Bomba Centrífuga':
                st.markdown("**Tipo Sugerido:** `Mass Flow Inlet` ou `Velocity Inlet`")
                st.markdown(f"- **Mass Flow Rate:** `{res['m_dot']:.3f} kg/s`\n- **Direction Specification:** Normal to Boundary\n- **Turbulence:** Intensity = 5%")
            else:
                st.markdown("**Tipo Sugerido:** `Pressure Inlet` ou `Mass Flow Inlet`")
                st.markdown(f"- **Pressure (Total):** `{(res['H_teo'] * 9.81 * rho):.0f} Pa` (Calculada da Carga)\n- OU **Mass Flow Rate:** `{res['m_dot']:.3f} kg/s`")
        with c2:
            st.markdown("### OUTLET")
            if maq_type == 'Bomba Centrífuga':
                st.markdown("**Tipo Sugerido:** `Pressure Outlet`")
                st.markdown(f"- **Gauge Pressure:** `0 Pa` (Referência para Delta P).\n- **Backflow Hydraulic Diameter:** `{(2 * res['b2']):.4f} m` (Aproximação).")
            else:
                st.markdown("**Tipo Sugerido:** `Pressure Outlet`")
                st.markdown("- **Gauge Pressure:** Depende da cota de restituição no tubo de sucção.")
                
        st.info("💡 **Dica Prática de Inicialização**: Inicialize o domínio de forma Standard a partir do Inlet para evitar Flying Point Exception.")
    else:
        st.info("Calcule a cinemática base primeiro.")

# ==========================================
# TAB 3: Malha & Turbulência (Y+)
# ==========================================
with tab3:
    if res.get('is_complete'):
        st.subheader("Dimensionamento da Primeira Camada de Malha (First Cell Height)")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            mod_turb = st.selectbox("Modelo de Turbulência Alvo", ["k-omega SST (Y+ < 1 até 5)", "k-epsilon Realizable (Y+ 30 a 300)"])
            is_sst = "SST" in mod_turb
            y_target = st.number_input("Y+ Desejado", value=1.0 if is_sst else 50.0, step=0.5)
            
            beta_med = (res.get('beta1',22) + res.get('beta2',25)) / 2
            L_est = (res.get('D2',0.3)/2 - res.get('D1',0.15)/2) / math.cos(math.radians(beta_med))
            LChar = st.number_input("Comps. Caract. (m) - Corda", value=float(f"{L_est:.4f}"), step=0.01)
            
        with c2:
            st.markdown("### Memória de Cálculo Dinâmica")
            W_med = (res['W_in'] + res['W_out']) / 2.0
            Re = (rho * W_med * LChar) / mu if (W_med > 0 and LChar > 0) else 0
            Cf = math.pow(2 * math.log10(max(Re, 1)) - 0.65, -2.3) if Re > 1 else 0
            Tau_w = 0.5 * Cf * rho * math.pow(W_med, 2)
            U_tau = math.sqrt(Tau_w / rho) if Tau_w > 0 else 0
            delta_y = (y_target * mu) / (rho * U_tau) if U_tau > 0 else 0
            
            st.write(f"- **Velocidade de Referência (W_med):** {W_med:.2f} m/s")
            st.write(f"- **Número de Reynolds (Re):** {Re:.2e}")
            st.write(f"- **Coef. de Atrito (Cf):** {Cf:.5f}")
            st.write(f"- **Tensão Cisalhamento (Tau_w):** {Tau_w:.2f} Pa")
            
            st.success(f"📏 **Altura da Primeira Célula (Δy):** {(delta_y * 1000):.5f} mm")
            
            if is_sst and y_target > 5:
                st.error("Para k-omega SST, não é recomendado usar Y+ > 5. Você perderá a resolução da subcamada viscosa.")
            elif not is_sst and y_target < 30:
                st.error("Modelos k-epsilon com Wall Functions padrão requerem a primeira célula na região logarítmica (Y+ > 30).")
    else:
        st.info("Calcule a cinemática base primeiro.")

# ==========================================
# TAB 4: Validação CFD
# ==========================================
with tab4:
    if res.get('is_complete'):
        st.subheader("Diagnóstico de CFD Pós-Processamento")
        st.markdown("Insira os valores extraídos dos Reports do Fluent para comparar com o modelo analítico.")
        
        c1, c2, c3 = st.columns(3)
        torque_cfd = c1.number_input("Torque Extraído (N.m)", value=float(f"{(res['Torque'] * 0.85):.2f}"))
        deltaP_cfd = c2.number_input("Delta P Total Extraído (Pa)", value=float(f"{(res['H_teo'] * rho * 9.81 * 0.8):.2f}"))
        beta2_cfd = c3.number_input("Ângulo Relativo Médio Saída β2 (°)", value=float(f"{(res['beta2'] - 3.0):.1f}"))
        
        if st.button("Analisar Desempenho", type="primary"):
            head_cfd = deltaP_cfd / (rho * 9.81) if rho > 0 else 0
            head_teo = res['H_teo']
            torque_teo = res['Torque']
            
            err_head = ((head_cfd - head_teo) / head_teo) * 100 if head_teo > 0 else 0
            err_torque = ((torque_cfd - torque_teo) / torque_teo) * 100 if torque_teo > 0 else 0
            slip = res['beta2'] - beta2_cfd
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Carga CFD vs Teórica", f"{head_cfd:.2f} m", f"{err_head:+.1f}%", delta_color="normal" if abs(err_head)<10 else "inverse")
            mc2.metric("Torque CFD vs Teórico", f"{torque_cfd:.2f} N.m", f"{err_torque:+.1f}%", delta_color="normal" if abs(err_torque)<10 else "inverse")
            mc3.metric("Slip Factor (Desvio)", f"{slip:.1f}°", "Ideal: 0°", delta_color="off")
            
            st.markdown("### Matriz de Inteligência e Correção")
            if slip > 2.0:
                st.warning(f"**Escorregamento (Slip) excessivo na saída ({slip:.1f}° de desvio):** Aumentar o número de pás (Z) ou utilizar lâminas de divisão (Splitter blades). Aumentar ligeiramente o β2.")
            if err_head < -15.0:
                st.error("**Perda severa de Carga/Eficiência:** Indício de Descolamento de Camada Limite (Stall). Verifique a curvatura da pá e reduza gradientes adversos.")
            if maq_type == 'Bomba Centrífuga':
                st.info("**Checagem de Cavitação:** Plote Contornos de Pressão Absoluta no bordo de ataque e compare com a pressão de vapor do fluido (~2500 Pa).")
            
    else:
        st.info("Calcule a cinemática base primeiro.")

# ==========================================
# TAB 5: Scripts TUI (Fluent)
# ==========================================
with tab5:
    if res.get('is_complete'):
        st.subheader("Configuração do Setup CFD (Text-User-Interface)")
        
        with st.form("fluent_tui_form2"):
            motion_type = st.radio("Selecione o Método", ["mrf", "mesh_motion"], format_func=lambda x: "Frame Motion (MRF - Steady)" if x == "mrf" else "Mesh Motion (Transient)")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Nomes das Zonas (Boundaries / Cell Zones)**")
                zone_inlet = st.text_input("Boundary de Entrada", value="inlet")
                zone_outlet = st.text_input("Boundary de Saída", value="outlet")
                zone_rotor = st.text_input("Wall do Rotor", value="rotor_wall")
                zone_interior = st.text_input("Cell Zone Rotor Fluid", value="interior_rotor")
                
            with c2:
                st.markdown("**Eixo de Rotação e Centro de Gravidade (CG)**")
                ax_x = st.number_input("Axis X", value=0.0)
                ax_y = st.number_input("Axis Y", value=0.0)
                ax_z = st.number_input("Axis Z", value=1.0)
                cg_x = st.number_input("CG X", value=0.0)
                cg_y = st.number_input("CG Y", value=0.0)
                cg_z = st.number_input("CG Z", value=0.0)
                
            c_bot1, c_bot2, c_bot3 = st.columns(3)
            with c_bot1:
                max_iterations = st.number_input("Máx. Iterações (Steady)", value=300)
            with c_bot2:
                num_time_steps = st.number_input("Passos de Tempo (Transient)", value=360)
                max_iter_per_step = st.number_input("Iterações por Passo", value=20)
            with c_bot3:
                residual_threshold = st.text_input("Critério de Convergência (Resíduos)", value="1e-4")
                
            gerar_btn = st.form_submit_button("Gerar Script TUI")
            
        if gerar_btn:
            rpm_val = res.get('N', 1750.0)
            
            script = f"; ====== Ansys Fluent TUI Setup Script ======\n"
            script += f"; Generated for {maq_type} | RPM: {rpm_val:.1f}\n"
            script += f"; Fluid Density: {rho} | Viscosity: {mu}\n\n"
            
            script += "; --- 1. General Settings ---\n"
            script += "/define/models/solver/density-based-implicit no\n"
            if motion_type == 'mrf':
                script += "/define/models/steady yes\n"
            else:
                script += "/define/models/unsteady-2nd-order yes\n"
    
            script += "\n; --- 2. Turbulence & Material ---\n"
            script += "/define/models/viscous/kw-sst yes\n"
            script += f"/define/materials/change-create fluid working_fluid yes constant {rho} no no yes constant {mu} no no no\n"
    
            script += "\n; --- 3. Boundary Conditions ---\n"
            if motion_type == 'mrf':
                script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid no yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm_val} no no no no no no\n"
            else:
                script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid yes yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm_val} no no no no no no\n"
    
            script += "\n; --- 4. Reports Definitions ---\n"
            script += f"/solve/report-definitions/add mflow-inlet surface-massflow surface-names {zone_inlet} () quit\n"
            script += f"/solve/report-definitions/add mflow-outlet surface-massflow surface-names {zone_outlet} () quit\n"
            script += f"/solve/report-definitions/add torque-rotor surface-moment moment-center {cg_x} {cg_y} {cg_z} moment-axis {ax_x} {ax_y} {ax_z} surface-names {zone_rotor} () quit\n"
            
            freq_type = 'iteration' if motion_type == 'mrf' else 'time-step'
            
            script += "\n; --- 5. Report Files ---\n"
            script += f"/solve/report-files/add mflow-inlet-file report-defs mflow-inlet () file-name mflow-inlet.out print yes frequency-of {freq_type} frequency 1 quit\n"
            script += f"/solve/report-files/add mflow-outlet-file report-defs mflow-outlet () file-name mflow-outlet.out print yes frequency-of {freq_type} frequency 1 quit\n"
            script += f"/solve/report-files/add torque-rotor-file report-defs torque-rotor () file-name torque-rotor.out print yes frequency-of {freq_type} frequency 1 quit\n"
            
            script += "\n; --- 6. Operating & Convergence ---\n"
            script += "/define/operating-conditions/operating-pressure 0\n"
            script += f"/solve/monitors/residual/convergence-criteria {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold}\n"
    
            script += "\n; --- 7. Initialization ---\n"
            script += "/solve/initialize/hyb-initialization\n"
    
            if motion_type == 'mrf':
                script += "\n; --- 8. Solver Run (MRF) ---\n"
                script += f"/solve/iterate {max_iterations}\n"
            else:
                script += "\n; --- 8. Solver Run (Transient) ---\n"
                dt_1deg = (1.0 / (6.0 * rpm_val)) if rpm_val > 0 else 0
                script += f"/solve/set/time-step {dt_1deg:.6f}\n"
                script += f"/solve/set/max-iterations-per-time-step {max_iter_per_step}\n"
                script += f"/solve/dual-time-iterate {num_time_steps} {max_iter_per_step}\n"
    
            script += "; End Setup\n"
            
            st.text_area("Copiar e colar no prompt do Fluent", value=script, height=350)
    else:
        st.info("Calcule a cinemática base primeiro.")


# ==========================================
# TAB 6: Assistente de IA
# ==========================================
with tab6:
    st.subheader("Assistente Especialista em CFD e Turbomáquinas")
    st.markdown("Descreva resultados ou faça perguntas sobre mecânica dos fluidos, cavitação ou turbulência.")
    
    # Simple Mock for AI as streamlit cannot easily do async APIs without actual endpoints
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ex: Estou observando grande recirculação na saída da pá, o que pode ser?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            st.markdown("Analisando com base nos seus parâmetros numéricos...")
            if res.get('is_complete'):
                rpm = res.get('N', 0)
                beta2 = res.get('beta2', 0)
                response = f"**Diagnóstico baseado nas medições e teoria:** A rotação está em {rpm:.0f} RPM e o ângulo de saída é {beta2:.1f}°. Se você nota recirculação, pode ser que o número de pás ({Z}) seja insuficiente causando \"Slip\" (escorregamento) excessivo, ou o gradiente de pressão está adverso devido à rápida difusão na voluta. Tente prolongar a corda e reduzir o carregamento por pá (aumentando Z)."
            else:
                response = "Por favor, conclua o dimensionamento cinemático na barra lateral primeiro para obtermos métricas de base."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
