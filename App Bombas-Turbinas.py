import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import google.generativeai as genai

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
            if alpha is not None:
                if abs(alpha - 90.0) < 1e-3:
                    set_val('Cu'+s_i, 0.0)
                else:
                    r = math.radians(alpha)
                    if Cm is not None:
                        if abs(math.sin(r)) > 1e-5: set_val('Cu'+s_i, Cm / math.tan(r))
                    if Cu is not None:
                        set_val('Cm'+s_i, Cu * math.tan(r))
                    if C is not None:
                        set_val('Cu'+s_i, C * math.cos(r))
                        set_val('Cm'+s_i, C * math.sin(r))

            if Cm is not None and Wu is not None and Wu != 0: set_val('beta'+s_i, math.degrees(math.atan2(Cm, Wu)))
            if beta is not None:
                if abs(beta - 90.0) < 1e-3:
                    set_val('Wu'+s_i, 0.0)
                else:
                    r = math.radians(beta)
                    if Cm is not None:
                        if abs(math.sin(r)) > 1e-5: set_val('Wu'+s_i, Cm / math.tan(r))
                    if Wu is not None:
                        set_val('Cm'+s_i, Wu * math.tan(r))
                    if W is not None:
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
    
    # Pre-calculate before rendering UI to allow placeholders to reflect the current state without lag
    _default_inputs = {
        'Q': 0.1, 'N': 1750.0, 'H_teo': None, 'Potencia': None,
        'D1': 0.15, 'D2': 0.3, 'b1': 0.04, 'b2': 0.02,
        'beta1': 22.0, 'beta2': 25.0, 'alpha1': 90.0, 'alpha2': None,
        'U1': None, 'Cm1': None, 'Cu1': None, 'Wu1': None, 'C1': None, 'W1': None,
        'U2': None, 'Cm2': None, 'Cu2': None, 'Wu2': None, 'C2': None, 'W2': None,
    }
    
    current_inputs = {}
    for k in _default_inputs.keys():
        widget_key = f"in_{k}"
        if widget_key in st.session_state:
            val_str = str(st.session_state[widget_key]).strip().replace(',', '.')
            try:
                current_inputs[k] = float(val_str) if val_str != "" else None
            except ValueError:
                current_inputs[k] = None
        else:
            current_inputs[k] = _default_inputs[k]

    res = solve_kinematic_system(maq_type, rho, current_inputs)
    
    def input_field(key, label):
        user_val = current_inputs.get(key)
        calc_val = res.get(key)
        ph = f"{calc_val:.4f} (calc)" if calc_val is not None else ""
        return st.text_input(label, value=str(user_val) if user_val is not None else "", placeholder=ph, key=f"in_{key}")

    c1, c2 = st.columns(2)
    with c1:
        Q = input_field('Q', 'Vazão Q (m³/s)')
        H_teo = input_field('H_teo', 'Carga (m)')
    with c2:
        N = input_field('N', 'Rotação (RPM)')
        Potencia = input_field('Potencia', 'Potência (W)')

    with st.expander("Estação 1", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            input_field('D1', 'D1')
            input_field('alpha1', 'alpha1')
            input_field('U1', 'U1')
            input_field('Cu1', 'Cu1')
            input_field('C1', 'C1')
        with c2:
            input_field('b1', 'b1')
            input_field('beta1', 'beta1')
            input_field('Cm1', 'Cm1')
            input_field('Wu1', 'Wu1')
            input_field('W1', 'W1')

    with st.expander("Estação 2", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            input_field('D2', 'D2')
            input_field('alpha2', 'alpha2')
            input_field('U2', 'U2')
            input_field('Cu2', 'Cu2')
            input_field('C2', 'C2')
        with c2:
            input_field('b2', 'b2')
            input_field('beta2', 'beta2')
            input_field('Cm2', 'Cm2')
            input_field('Wu2', 'Wu2')
            input_field('W2', 'W2')


# ==========================================
# Tabs definition
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Cinemática & Triângulos", 
    "Setup CFD (Manual)",
    "Malha & Turbulência (Y+)",
    "Validação & Diagnóstico",
    "Scripts TUI (Fluent)",
    "Assistente de IA",
    "Geometria (SolidWorks)",
    "Malha (Meshing)"
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

        st.markdown("---")
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

# ==========================================


# ==========================================
# TAB 7: Geometria (SolidWorks)
# ==========================================
with tab7:
# GEOMETRIA (SolidWorks)
# ==========================================
    st.markdown("---")
    st.subheader("Geração de Script VBA (SolidWorks)")
    st.markdown("Automatize a criação da geometria 3D utilizando a API do SolidWorks. Copie o script abaixo e rode como Macro no SolidWorks.")
    
    sw_option = st.radio("O que deseja construir?", ["Apenas Rotor", "Rotor + Domínio Fluido (CFD)"])
    
    maq = maq_type
    # As variáveis já estão calculadas acima, mas garantindo aqui
    D1 = res.get('D1', 0)
    D2 = res.get('D2', 0)
    b1 = res.get('b1', 0)
    b2 = res.get('b2', 0)
    Z = res.get('Z', 6)
    
    sw_script = f"""' ==============================================================================
' EXPERIMENTAL SOLIDWORKS VBA MACRO - AUTOGENERATED (REVISADO)
' Configuration: {sw_option}
' Machine: {maq}
' Z (Pás) = {Z}
' ==============================================================================
Dim swApp As Object
Dim Part As Object
Dim boolstatus As Boolean
Dim defaultTemplate As String
Dim longstatus As Long, longwarnings As Long

Sub main()
    Set swApp = Application.SldWorks
    Set Part = swApp.ActiveDoc
    
    ' Busca o template padrão do usuário
    If Part Is Nothing Then
    defaultTemplate = swApp.GetUserPreferenceStringValue(8) ' 8 = swDefaultTemplatePart
    Set Part = swApp.NewDocument(defaultTemplate, 0, 0, 0)
    End If
    
    ' Seleciona o Plano Frontal e abre o esboço
    boolstatus = Part.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, False, 0, Nothing, 0)
    Part.SketchManager.InsertSketch True
    Part.ClearSelection2 True
    
    ' Criando eixo de revolução (Eixo Y)
    Dim swCenterLine As Object
    Set swCenterLine = Part.SketchManager.CreateCenterLine(0, -1, 0, 0, 1, 0)
    
    ' Perfil Meridional (Hub e Shroud)
    Dim D1 As Double, D2 As Double, b1 As Double, b2 As Double
    D1 = {D1:.5f} : D2 = {D2:.5f}
    b1 = {b1:.5f} : b2 = {b2:.5f}
    
    ' Desenhando um perfil FECHADO para gerar um sólido
    ' Linha Superior (Shroud)
    Part.SketchManager.CreateLine D1 / 2, b1 / 2, 0, D2 / 2, b2 / 2, 0
    ' Linha de Saída (Fechamento vertical em D2)
    Part.SketchManager.CreateLine D2 / 2, b2 / 2, 0, D2 / 2, -b2 / 2, 0
    ' Linha Inferior (Hub)
    Part.SketchManager.CreateLine D2 / 2, -b2 / 2, 0, D1 / 2, -b1 / 2, 0
    ' Linha de Entrada (Fechamento vertical em D1)
    Part.SketchManager.CreateLine D1 / 2, -b1 / 2, 0, D1 / 2, b1 / 2, 0
    
    ' Seleciona a linha de centro para ser o eixo de revolução
    boolstatus = Part.Extension.SelectByID2("Line1", "SKETCHSEGMENT", 0, 0, 0, False, 16, Nothing, 0)
    
    ' Revolução do cubo/Hub (2 * Pi rad)
    Part.FeatureManager.FeatureRevolve2 True, True, False, False, False, False, 0, 0, 6.283185307, 0, False, False, 0.01, 0.01, 0, 0, 0, True, True, True
"""
    if "Domínio Fluido" in sw_option:
        sw_script += f"""
    ' Criando Enclosure (Domínio Fluido Exterior)
    ' Configurado para simulação no CFD (Inlet estendido e Voluta simplificada)
    Part.Extension.SelectByID2 "Front Plane", "PLANE", 0, 0, 0, False, 0, Nothing, 0
    Part.SketchManager.InsertSketch True
    Part.SketchManager.CreateCircleByRadius 0, 0, 0, D2 * 1.5
    Part.FeatureManager.FeatureExtrusion2 True, False, False, 0, 0, b2 * 3, 0.01, False, False, False, False, 0, 0, False, False, False, False, True, True, True, 0, 0, False
"""
    sw_script += """
    swApp.SendMsgToUser "Geometria gerada com sucesso! Verifique e ajuste as pás e a voluta se necessário."
End Sub
"""
    st.code(sw_script, language="vba")

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
            y_target = st.number_input("Y+ Desejado", value=1.0 if is_sst else 50.0, step=0.5, key="y_target_input")
            
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
        
        col_setup, col_script = st.columns([1, 1])
        
        with col_setup:
            st.markdown("### 1. Selecionar Método")
            motion_type = st.radio("Método", ["mrf", "mesh_motion"], format_func=lambda x: "Frame Motion (MRF - Steady)" if x == "mrf" else "Mesh Motion (Transient)", label_visibility="collapsed")
            
            st.markdown("### 2. Nomes das Zonas (Boundaries / Cell Zones)")
            cz1, cz2 = st.columns(2)
            with cz1:
                zone_inlet = st.text_input("Inlet (Entrada)", value="inlet")
                zone_rotor = st.text_input("Parede do Rotor", value="rotor")
            with cz2:
                zone_outlet = st.text_input("Outlet (Saída)", value="outlet")
                zone_stator = st.text_input("Domínio Ext. (Stator)", value="interior_stator")
                
            zone_interior = st.text_input("Domínio Int. (Rotor Fluid)", value="interior_rotor")
                
            st.markdown("### 3. Eixo de Rotação e CG")
            c_cg1, c_cg2, c_cg3 = st.columns(3)
            with c_cg1:
                cg_x = st.number_input("CG X", value=0.0)
                ax_x = st.number_input("Eixo X", value=0.0)
            with c_cg2:
                cg_y = st.number_input("CG Y", value=0.0)
                ax_y = st.number_input("Eixo Y", value=0.0)
            with c_cg3:
                cg_z = st.number_input("CG Z", value=0.0)
                ax_z = st.number_input("Eixo Z", value=1.0)
                
            if motion_type == 'mrf':
                st.markdown("### 4. Configuração MRF (Steady)")
                max_iterations = st.number_input("Número Máximo de Iterações", value=300)
            else:
                st.markdown("### 4. Configuração Transient (CFL)")
                c_trans1, c_trans2 = st.columns(2)
                with c_trans1:
                    num_time_steps = st.number_input("Passos de Tempo (Time-Steps)", value=360)
                    courant = st.number_input("Número de Courant (Co)", value=1.0)
                with c_trans2:
                    max_iter_per_step = st.number_input("Iterações p/ Time-Step", value=20)
                    min_mesh = st.number_input("Tamanho da Menor Malha (dx em metros)", value=0.005, format="%.4f")
                
                rpm_val = res.get('N', 1750.0)
                W_med = (res.get('W_in', 0) + res.get('W_out', 0)) / 2.0
                dt_1deg = (1.0 / (6.0 * rpm_val)) if rpm_val > 0 else 0
                dt_cfl = (courant * min_mesh) / W_med if W_med > 0 else 0
                
                st.info(f"**Velocidade de Ref. (W_med):** {W_med:.2f} m/s\n\n**Passo de Tempo Sugerido (1 grau):** {dt_1deg:.3e} s\n\n**Passo de Tempo Sugerido (Courant={courant}):** {dt_cfl:.3e} s")
                
            st.markdown("### 5. Critérios de Convergência")
            residual_threshold = st.text_input("Resíduo Permitido (Ex: 1e-4, 1e-5)", value="1e-4")
            
            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                chk_cont = st.checkbox("Continuidade", value=True)
            with c_res2:
                chk_vel = st.checkbox("Velocidades (x, y, z)", value=True)
            with c_res3:
                chk_turb = st.checkbox("Turbulência (k, ω/ε)", value=True)
                
        # Generate TUI Script based on the reactive inputs
        rpm_val = res.get('N', 1750.0)
        rad_s_val = rpm_val * math.pi / 30.0
        
        script = f"; ====== Ansys Fluent TUI Setup Script ======\n"
        script += f"; Generated automatically based on Turbomachinery parameters\n"
        script += f"; Machine: {maq_type} | RPM: {rpm_val:.1f} | Rad/s: {rad_s_val:.4f}\n"
        script += f"; Fluid Density: {rho} | Viscosity: {mu}\n\n"
        
        script += "; --- 1. General Settings ---\n"
        script += "/define/models/solver/pressure-based yes\n"
        if motion_type == 'mrf':
            script += "/define/models/steady yes\n"
        else:
            script += "/define/models/unsteady-2nd-order yes\n"

        script += "\n; --- 2. Turbulence Model ---\n"
        if 'mod_turb' in locals() and "SST" in mod_turb:
            script += "/define/models/viscous/kw-sst yes\n"
        else:
            script += "/define/models/viscous/ke-realizable yes\n"
            
        script += "\n; --- 3. Materials Setup ---\n"
        script += "; Defines working fluid\n"
        script += f"/define/materials/change-create air working_fluid yes constant {rho} no no yes constant {mu} no no no\n"

        script += "\n; --- 4. Boundary Conditions & Zones ---\n"
        script += "; Note: Change zone names below if they do not match your mesh\n"
        script += "; Assign fluid material to domains\n"
        script += f"/define/boundary-conditions/fluid {zone_stator} yes working_fluid no no no no 0 no 0 no 0 no 0 no 1 no no no no\n"
        
        if motion_type == 'mrf':
            script += "; Setup MRF (Frame Motion) on Rotor interior\n"
            script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid no yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rad_s_val:.5f} no no no no no no\n"
        else:
            script += "; Setup Mesh Motion on Rotor interior\n"
            script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid yes yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rad_s_val:.5f} no no no no no no\n"

        script += "; Setup Inlet and Outlet\n"
        m_dot = res.get('m_dot', 0)
        p_total = res.get('H_teo', 0) * 9.81 * rho
        if maq_type == 'Bomba Centrífuga':
            script += f"; Configure Inlet (Mass Flow Rate = {m_dot:.3f} kg/s)\n"
            script += f"/define/boundary-conditions/modify-zones/zone-type {zone_inlet} mass-flow-inlet\n"
            script += f"/define/boundary-conditions/set/mass-flow-inlet {zone_inlet} () mass-flow no {m_dot:.5f} quit\n"
            script += f"; Configure Outlet (Pressure = 0 Pa)\n"
            script += f"/define/boundary-conditions/modify-zones/zone-type {zone_outlet} pressure-outlet\n"
            script += f"/define/boundary-conditions/set/pressure-outlet {zone_outlet} () gauge-pressure yes 0 quit\n"
        else:
            script += f"; Configure Inlet (Pressure Inlet = {p_total:.0f} Pa)\n"
            script += f"/define/boundary-conditions/modify-zones/zone-type {zone_inlet} pressure-inlet\n"
            script += f"/define/boundary-conditions/set/pressure-inlet {zone_inlet} () gauge-total-pressure yes {p_total:.0f} quit\n"
            script += f"; Configure Outlet (Pressure = 0 Pa)\n"
            script += f"/define/boundary-conditions/modify-zones/zone-type {zone_outlet} pressure-outlet\n"
            script += f"/define/boundary-conditions/set/pressure-outlet {zone_outlet} () gauge-pressure yes 0 quit\n"

        script += "\n; --- 5. Reports Definitions ---\n"
        script += "; Inlet and Outlet Mass Flows\n"
        script += f"/solve/report-definitions/add mflow-inlet surface-massflow surface-names {zone_inlet} () quit\n"
        script += f"/solve/report-definitions/add mflow-outlet surface-massflow surface-names {zone_outlet} () quit\n"
        script += "; Rotor Torque\n"
        script += f"/solve/report-definitions/add torque-rotor surface-moment moment-center {cg_x} {cg_y} {cg_z} moment-axis {ax_x} {ax_y} {ax_z} surface-names {zone_rotor} () quit\n"
        
        freq_type = 'iteration' if motion_type == 'mrf' else 'time-step'
        
        script += "\n; --- 6. Report Files ---\n"
        script += f"/solve/report-files/add mflow-inlet-file report-defs mflow-inlet () file-name mflow-inlet.out print yes frequency-of {freq_type} frequency 1 quit\n"
        script += f"/solve/report-files/add mflow-outlet-file report-defs mflow-outlet () file-name mflow-outlet.out print yes frequency-of {freq_type} frequency 1 quit\n"
        script += f"/solve/report-files/add torque-rotor-file report-defs torque-rotor () file-name torque-rotor.out print yes frequency-of {freq_type} frequency 1 quit\n"
        
        script += "\n; --- 7. Operating & Convergence ---\n"
        script += "/define/operating-conditions/operating-pressure 0\n"
        
        # Build residual criteria string based on checks
        try:
            rt_val = float(residual_threshold)
            rt_str = str(residual_threshold)
        except ValueError:
            rt_str = "1e-4"
            
        script += "; Set residuals for active equations\n"
        # The exact TUI command for residuals depends on the exact equations solved, normally it's:
        # /solve/monitors/residual/convergence-criteria <continuity> <x-vel> <y-vel> <z-vel> <k> <omega> ...
        # But a safer approach is to set them individually for the enabled ones if possible, but Fluent CLI 
        # usually wants a list. Since we don't know the exact order without checking models, we just provide
        # standard 6 values for 3D k-omega/k-epsilon.
        script += f"/solve/monitors/residual/convergence-criteria {rt_str} {rt_str} {rt_str} {rt_str} {rt_str} {rt_str}\n"

        script += "\n; --- 8. Initialization ---\n"
        script += "/solve/initialize/hyb-initialization\n"

        if motion_type == 'mrf':
            script += "\n; --- 9. Solver Run (MRF) ---\n"
            script += f"/solve/iterate {max_iterations}\n"
        else:
            script += "\n; --- 9. Solver Run (Transient) ---\n"
            script += f"; Courant CFL = {courant} | Min Mesh = {min_mesh} | Reference W = {W_med:.2f}\n"
            script += f"/solve/set/time-step {dt_cfl:.6f}\n"
            script += f"/solve/set/max-iterations-per-time-step {max_iter_per_step}\n"
            script += f"/solve/dual-time-iterate {num_time_steps} {max_iter_per_step}\n"

        script += "; End Setup\n"
        
        with col_script:
            c_header_1, c_header_2 = st.columns([1, 1])
            c_header_1.markdown("### TUI Console Output")
            # We don't really have a copy to clipboard button in pure streamlit unless we use st.code which has it built-in.
            # st.code has a copy button on the top right
            st.code(script, language="scheme")
    else:
        st.info("Calcule a cinemática base primeiro.")



# ==========================================
# TAB 6: Assistente de IA

# ==========================================
with tab6:
    st.subheader("Assistente Especialista em CFD e Turbomáquinas")
    st.markdown("Descreva resultados ou faça perguntas sobre mecânica dos fluidos, cavitação ou turbulência.")
    
    user_api_key = st.text_input("Gemini API Key (Obrigatório)", type="password", help="Insira sua Chave de API do Gemini para habilitar o assistente (não é salva e processada localmente nesta sessão).")
    gemini_key = os.environ.get("GEMINI_API_KEY") or user_api_key
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    else:
        gemini_model = None
    
    if gemini_model is None:
        st.warning("⚠️ Insira sua Chave de API do Gemini acima para habilitar o Assistente de IA.")
    
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
            if gemini_model:
                with st.spinner("Analisando com base nos seus parâmetros numéricos..."):
                    try:
                        context = f"Máquina: {maq_type}\nFluido: Rho = {rho} kg/m³ | Mu = {mu} Pa.s\nNúmero de Pás: {Z}\n"
                        if res.get('is_complete'):
                            context += f"Carga Teórica: {res.get('H_teo', 0):.2f} m | Potência: {res.get('Potencia', 0)/1000:.2f} kW | Rotação: {res.get('N', 0):.1f} RPM\n"
                            context += f"Entrada (U, C, W, beta, alpha): {res.get('U_in',0):.1f}, {res.get('C_in',0):.1f}, {res.get('W_in',0):.1f}, {res.get('beta_in',0):.1f}°, {res.get('alpha_in',0):.1f}°\n"
                            context += f"Saída (U, C, W, beta, alpha): {res.get('U_out',0):.1f}, {res.get('C_out',0):.1f}, {res.get('W_out',0):.1f}, {res.get('beta_out',0):.1f}°, {res.get('alpha_out',0):.1f}°\n"
                        else:
                            context += "Projeto cinemático ainda não concluído pelo usuário.\n"
                        
                        system_prompt = f"Você é um engenheiro especialista em CFD e Turbomáquinas. Contexto atual do projeto do usuário:\n{context}\nResponda agora a seguinte pergunta ou observação do usuário. Seja objetivo, analítico e proponha soluções técnicas viáveis (como alterar Z, corda, beta2, malha, etc.):\n{prompt}"
                        
                        history = []
                        for m in st.session_state.messages[:-1]:
                            role = 'user' if m['role'] == 'user' else 'model'
                            history.append({"role": role, "parts": [m['content']]})
                            
                        chat = gemini_model.start_chat(history=history)
                        response = chat.send_message(system_prompt).text
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        err_msg = f"❌ Erro ao acessar modelo Gemini: {str(e)}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                fallback_msg = "O modelo de IA não está inicializado devido à falta da respectiva Chave de API."
                st.markdown(fallback_msg)
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})



# ==========================================
# TAB 8: Malha (Meshing)
# ==========================================
with tab8:
    st.subheader("Geração de Script (Meshing)")
    st.markdown("Scripts automatizados para Ansys Meshing ou Fluent Meshing utilizando a altura da primeira camada (Y+) calculada.")
    
    meshTool = st.radio("Ferramenta de Meshing", ["Ansys Meshing (Python Snippet)", "Fluent Meshing (TUI)"])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        zoneInlet = st.text_input("Zone Inlet", value="inlet")
    with c2:
        zoneOutlet = st.text_input("Zone Outlet", value="outlet")
    with c3:
        zoneRotor = st.text_input("Zone Rotor", value="rotor")
    
    y_plus = st.session_state.get('y_target_input', 1.0)
    
    W_med = (res.get('W_in', 0) + res.get('W_out', 0)) / 2.0
    LChar_calc = res.get('D2', 0)
    mu_calc = mu
    rho_calc = rho
    Re_calc = (rho_calc * W_med * LChar_calc) / mu_calc if mu_calc > 0 else 0
    Cf_calc = (2 * math.log10(Re_calc) - 0.65) ** -2.3 if Re_calc > 1 else 0
    Tau_w_calc = 0.5 * Cf_calc * rho_calc * (W_med**2)
    U_tau_calc = math.sqrt(Tau_w_calc / rho_calc) if Tau_w_calc > 0 else 0
    
    yHeight = 0
    if U_tau_calc > 0:
        yHeight = (y_plus * mu_calc) / (rho_calc * U_tau_calc)

    # Fallback caso seja 0
    if yHeight <= 0:
        yHeight = 1.75110e-6

    yExp = f"{yHeight:.5e}"

    if meshTool == "Ansys Meshing (Python Snippet)":
        meshScript = f"""# ==============================================================================
# ANSYS MESHING PYTHON SNIPPET - AUTOGENERATED
# Configurado para Ansys Student | y+ = {y_plus}
# ==============================================================================

# Acessar a malha do modelo ativo
mesh = Model.Mesh

# Configurações Globais de Tamanho
mesh.UseAdvancedSizeFunction = AdvancedSizeFunctionType.Curvature
mesh.CurvatureNormalAngle = Quantity(18.0, "deg") # Ângulo grosseiro para poupar nós
mesh.MinimumSize = Quantity(0.005, "m")
mesh.MaximumSize = Quantity(0.050, "m")

# --- Criação de Named Selections ---
# Nota: Em automação real, os IDs das faces mudam. 
# O ideal é que o SolidWorks exporte as faces já com cores ou atributos específicos
# para o script do Ansys procurar automaticamente. Aqui criamos os grupos vazios:
ns_inlet = Model.AddNamedSelection()
ns_inlet.Name = "{zoneInlet}"

ns_outlet = Model.AddNamedSelection()
ns_outlet.Name = "{zoneOutlet}"

ns_rotor = Model.AddNamedSelection()
ns_rotor.Name = "{zoneRotor}"

# --- Configuração da Camada Limite (Inflation) ---
inflation = mesh.AddInflation()
# inflation.Location = ... (Aqui entrariam os corpos/faces do domínio)
# inflation.BoundaryLocation = ... (Aqui entrariam as faces da pá/rotor)

inflation.InflationOption = 1 # 1 = First Layer Thickness
inflation.FirstLayerHeight = Quantity({yExp}, "m")
inflation.MaximumLayers = 15  # 15 camadas para transição suave do y+
inflation.GrowthRate = 1.2

# Gerar Malha
mesh.GenerateMesh()
"""
    else:
        meshScript = f"""; ====== Fluent Meshing TUI Workflow ======
; Configurado para Ansys Student (Redução de Nós) e Y+ = {y_plus}
/file/import/cad "rotational_domain.step"

; Nomeando Faces (se ainda nao vieram nomeadas do CAD)
; No Fluent Meshing a renomeação é feita via /boundary/manage/name
/boundary/manage/name "in" "{zoneInlet}"
/boundary/manage/name "out" "{zoneOutlet}"
/boundary/manage/name "walls" "{zoneRotor}"

; Define Curvature & Proximity Sizing
/size-functions/create curvature-size "curvature" 0.005 0.05 1.2 18
/size-functions/compute

; Gera Malha de Superficie
/mesh/surface/auto-mesh (*)

; -- Configuração da Camada Limite (Inflation / Prism Layers) --
; A geração de prismas no Fluent Meshing usa o menu /mesh/prism/
/mesh/prism/controls/first-height {yExp}
/mesh/prism/controls/growth-rate 1.2
/mesh/prism/controls/number-of-layers 15
/mesh/prism/create ("{zoneRotor}")

; Define Volume Hexcore pra economizar elementos
/mesh/auto-mesh/volume-mesh poly-hexcore

; Check Quality
/mesh/check-quality
/file/write-mesh "rotational_mesh.msh"
"""

    st.code(meshScript, language="python" if meshTool == "Ansys Meshing (Python Snippet)" else "scheme")
