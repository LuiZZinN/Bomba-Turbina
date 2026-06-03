import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Sistema de Turbomáquinas & CFD", layout="wide")

# ==========================================
# 1. Kinematic Solver
# ==========================================
def solve_kinematic_system(maq, rho, inputs):
    v = inputs.copy()
    changed = True
    iters = 0

    def set_val(k, val):
        nonlocal changed
        if v.get(k) is None or math.isnan(v[k]):
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
        if maq == 'Bomba Centrífuga':
            v['H_teo'] = (v['U2']*v['Cu2'] - v['U1']*v['Cu1']) / 9.81
            v['Torque'] = mdot * abs(v['U2']*v['Cu2'] - v['U1']*v['Cu1']) / v['omega']
        else:
            v['H_teo'] = (v['U1']*v['Cu1'] - v['U2']*v['Cu2']) / 9.81
            v['Torque'] = mdot * abs(v['U1']*v['Cu1'] - v['U2']*v['Cu2']) / v['omega']
        v['Potencia'] = v['Torque'] * v['omega']

    return v

# ==========================================
# Application State & Title
# ==========================================
st.title("Turbomáquinas - CFD Solver & Dimensionamento")

# ==========================================
# Sidebar: Setup Base
# ==========================================
st.sidebar.header("Parâmetros Gerais")
maq_type = st.sidebar.selectbox("Tipo de Máquina", ["Bomba Centrífuga", "Turbina Hidráulica"])
rho = st.sidebar.number_input("Densidade do Fluido (kg/m³)", value=998.0)
mu = st.sidebar.number_input("Viscosidade (kg/(m·s))", value=0.001, format="%.4f")

# ==========================================
# Tabs definition
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Cinemática", 
    "Triângulo de Vel.", 
    "Visualização 3D", 
    "Mapas de Desempenho", 
    "Diagnóstico", 
    "Scripts TUI (Fluent)"
])

# ==========================================
# TAB 1: Cinemática (Entradas e Solver)
# ==========================================
with tab1:
    st.subheader("Entradas do Sistema Cinemático")
    col1, col2, col3, col4 = st.columns(4)
    
    inputs = {}
    
    def input_col(col, label, key, default):
        val = col.text_input(label, value=str(default), key=key)
        try:
            return float(val) if val.strip() != "" else None
        except ValueError:
            return None

    inputs['Q'] = input_col(col1, "Vazão (Q)", "in_Q", 0.1)
    inputs['N'] = input_col(col1, "Rotação (RPM)", "in_N", 1750.0)
    inputs['D1'] = input_col(col1, "Diâmetro 1 (D1)", "in_D1", 0.15)
    inputs['b1'] = input_col(col1, "Largura 1 (b1)", "in_b1", 0.04)

    inputs['beta1'] = input_col(col2, "Beta 1 (deg)", "in_beta1", 22.0)
    inputs['alpha1'] = input_col(col2, "Alpha 1 (deg)", "in_alpha1", 90.0)
    inputs['D2'] = input_col(col2, "Diâmetro 2 (D2)", "in_D2", 0.3)
    inputs['b2'] = input_col(col2, "Largura 2 (b2)", "in_b2", 0.02)

    inputs['beta2'] = input_col(col3, "Beta 2 (deg)", "in_beta2", 25.0)
    inputs['alpha2'] = input_col(col3, "Alpha 2 (deg)", "in_alpha2", "")
    inputs['U1'] = input_col(col3, "Vel. Periférica 1 (U1)", "in_U1", "")
    inputs['Cu1'] = input_col(col3, "Tangencial Abs. 1 (Cu1)", "in_Cu1", "")

    inputs['Cm1'] = input_col(col4, "Meridional 1 (Cm1)", "in_Cm1", "")
    inputs['U2'] = input_col(col4, "Vel. Periférica 2 (U2)", "in_U2", "")
    inputs['Cu2'] = input_col(col4, "Tangencial Abs. 2 (Cu2)", "in_Cu2", "")
    inputs['Cm2'] = input_col(col4, "Meridional 2 (Cm2)", "in_Cm2", "")

    valid_inputs = {k: v for k, v in inputs.items() if v is not None}
    
    if st.button("Resolver Sistema", type="primary"):
        res = solve_kinematic_system(maq_type, rho, valid_inputs)
        st.session_state['res'] = res
        st.success("Calculado!")
        st.json(res)

if 'res' not in st.session_state:
    st.session_state['res'] = solve_kinematic_system(maq_type, rho, valid_inputs)

res = st.session_state['res']

# ==========================================
# TAB 2: Triângulos de Velocidade
# ==========================================
with tab2:
    st.subheader("Triângulo de Velocidades")
    
    if not res.get('is_complete'):
        st.warning("Variáveis insuficientes para plotar os triângulos. Volte na Cinemática.")
    else:
        def create_triangle_plot(U, Cu, Cm, W_u, title):
            fig = go.Figure()

            # Vector U (Periférica)
            fig.add_trace(go.Scatter(x=[0, U], y=[0, 0], mode='lines+text', name='U (Periférica)',
                                     line=dict(color='green', width=3, dash='dash')))

            # Vector C (Absoluta)
            fig.add_trace(go.Scatter(x=[0, Cu], y=[0, Cm], mode='lines+text', name='C (Absoluta)',
                                     line=dict(color='blue', width=3)))

            # Vector W (Relativa)
            fig.add_trace(go.Scatter(x=[U, Cu], y=[0, Cm], mode='lines+text', name='W (Relativa)',
                                     line=dict(color='red', width=3)))

            fig.update_layout(title=title, xaxis_title="Tangencial", yaxis_title="Meridional / Radial", 
                              yaxis=dict(scaleanchor="x", scaleratio=1))
            return fig

        c1, c2 = st.columns(2)
        with c1:
            try:
                fig1 = create_triangle_plot(res['U1'], res['Cu1'], res['Cm1'], res['Wu1'], "Estação 1 (Entrada)")
                st.plotly_chart(fig1, use_container_width=True)
            except:
                st.error("Erro ao plotar entrada")
        with c2:
            try:
                fig2 = create_triangle_plot(res['U2'], res['Cu2'], res['Cm2'], res['Wu2'], "Estação 2 (Saída)")
                st.plotly_chart(fig2, use_container_width=True)
            except:
                st.error("Erro ao plotar saída")

# ==========================================
# TAB 3: Visualização 3D do Rotor
# ==========================================
with tab3:
    st.subheader("Visualização 3D")
    
    if res.get('D1') and res.get('D2') and res.get('b1') and res.get('b2'):
        
        d1, d2 = res['D1'], res['D2']
        b1, b2 = res['b1'], res['b2']
        
        theta = np.linspace(0, 2*np.pi, 50)
        z_out = np.linspace(0, b2, 10)
        theta_grid, z_grid = np.meshgrid(theta, z_out)
        
        x_out = (d2 / 2) * np.cos(theta_grid)
        y_out = (d2 / 2) * np.sin(theta_grid)
        
        fig3d = go.Figure(data=[go.Surface(x=x_out, y=y_out, z=z_grid, colorscale='Blues', opacity=0.8)])
        
        # Inner surface
        z_in = np.linspace(0, b1, 10)
        theta_grid_in, z_grid_in = np.meshgrid(theta, z_in)
        x_in = (d1 / 2) * np.cos(theta_grid_in)
        y_in = (d1 / 2) * np.sin(theta_grid_in)
        
        fig3d.add_trace(go.Surface(x=x_in, y=y_in, z=z_grid_in, colorscale='Reds', opacity=0.8))
        
        fig3d.update_layout(title="Rotor - Perfil Simplificado", 
                            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Largura Z (m)'),
                            margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("São necessários os diâmetros (D1, D2) e larguras (b1, b2) para a visualização 3D.")

# ==========================================
# TAB 4: Mapas de Desempenho
# ==========================================
with tab4:
    st.subheader("Cálculo Off-Design & Curvas de Desempenho")
    
    if res.get('is_complete'):
        qs = np.linspace(0.5 * res['Q'], 1.5 * res['Q'], 50)
        
        h_array = []
        p_array = []
        
        omega = res['omega']
        beta2 = math.radians(res.get('beta2', 25))
        d2 = res['D2']
        b2 = res['b2']
        u2 = res['U2']
        
        for q_test in qs:
            cm2_test = q_test / (math.pi * d2 * b2)
            wu2_test = cm2_test / math.tan(beta2)
            cu2_test = u2 - wu2_test
            
            h_test = (u2 * cu2_test) / 9.81
            h_array.append(h_test)
            
            p_test = rho * q_test * 9.81 * h_test
            p_array.append(p_test)
            
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=qs, y=h_array, mode='lines', name='Head (H)'))
        fig_perf.update_layout(title='Curva H-Q (Teórica)', xaxis_title='Vazão Q (m³/s)', yaxis_title='Head (m)')
        st.plotly_chart(fig_perf)
        
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=qs, y=p_array, mode='lines', name='Potência (W)', line=dict(color='red')))
        fig_power.update_layout(title='Curva P-Q (Teórica)', xaxis_title='Vazão Q (m³/s)', yaxis_title='Potência (W)')
        st.plotly_chart(fig_power)
    else:
        st.info("Calcule a cinemática base primeiro.")

# ==========================================
# TAB 5: Diagnóstico
# ==========================================
with tab5:
    st.subheader("Diagnóstico e Validação do Projeto")
    
    if res.get('is_complete'):
        st.markdown("### Verificações Comuns")
        
        if res.get('Cm1') and res['Cm1'] > 15:
            st.error("Velocidade Meridional de entrada alta (> 15 m/s). Possível risco de cavitação severa se for líquido.")
        else:
            st.success("Velocidade Meridional de entrada aceitável.")
            
        if res.get('beta1') and not (15 <= res['beta1'] <= 40):
            st.warning(f"Ângulo Beta 1 ({res['beta1']:.1f}°) está fora da zona comum de projeto (15° - 40°).")
        else:
            st.success("Ângulo Beta 1 estruturalmente estável.")
            
        if res.get('beta2') and not (15 <= res['beta2'] <= 40):
            st.warning(f"Ângulo Beta 2 ({res['beta2']:.1f}°) está fora da zona comum (15° - 40°).")
        else:
            st.success("Ângulo Beta 2 estável.")
            
    else:
        st.info("Parâmetros incompletos para diagnóstico.")

# ==========================================
# TAB 6: Scripts TUI (Fluent)
# ==========================================
with tab6:
    st.subheader("Gerador de Script TUI para Ansys Fluent")
    
    with st.form("fluent_tui_form"):
        motion_type = st.radio("Método de Rotação", ["mrf", "mesh_motion"], format_func=lambda x: "Frame Motion (MRF - Steady)" if x == "mrf" else "Mesh Motion (Transient)")
        
        c1, c2 = st.columns(2)
        with c1:
            zone_inlet = st.text_input("Boundary de Entrada", value="inlet")
            zone_outlet = st.text_input("Boundary de Saída", value="outlet")
            zone_rotor = st.text_input("Wall do Rotor", value="rotor_wall")
            zone_interior = st.text_input("Cell Zone Rotor", value="interior_rotor")
            
        with c2:
            st.write("Eixo de Rotação")
            ax_x = st.number_input("Axis X", value=0.0)
            ax_y = st.number_input("Axis Y", value=0.0)
            ax_z = st.number_input("Axis Z", value=1.0)
            
            cg_x = st.number_input("CG X", value=0.0)
            cg_y = st.number_input("CG Y", value=0.0)
            cg_z = st.number_input("CG Z", value=0.0)
            
        if motion_type == "mrf":
            max_iterations = st.number_input("Máx. Iterações (Steady)", value=300)
        else:
            num_time_steps = st.number_input("Passos de Tempo (Transient)", value=360)
            max_iter_per_step = st.number_input("Iterações por Passo", value=20)
            courant = st.number_input("Courant Base", value=1.0)
            min_mesh = st.number_input("Delta X da Malha (min)", value=0.005, format="%.4f")
            
        residual_threshold = st.text_input("Critério de Resíduo", value="1e-4")
        mod_turb = st.selectbox("Modelo de Turbulência", ["SST", "Realizable"])
        
        gerar_btn = st.form_submit_button("Gerar Script TUI")
        
    if gerar_btn:
        rpm = res.get('N', 1750.0)
        
        script = f"; ====== Ansys Fluent TUI Setup Script ======\n"
        script += f"; Generated for {maq_type} | RPM: {rpm:.1f}\n"
        script += f"; Fluid Density: {rho} | Viscosity: {mu}\n\n"
        
        script += "; --- 1. General Settings ---\n"
        script += "/define/models/solver/density-based-implicit no\n"
        if motion_type == 'mrf':
            script += "/define/models/steady yes\n"
        else:
            script += "/define/models/unsteady-2nd-order yes\n"

        script += "\n; --- 2. Turbulence & Material ---\n"
        if mod_turb == 'SST':
            script += "/define/models/viscous/kw-sst yes\n"
        else:
            script += "/define/models/viscous/ke-realizable yes\n"
            
        script += f"/define/materials/change-create fluid working_fluid yes constant {rho} no no yes constant {mu} no no no\n"

        script += "\n; --- 3. Boundary Conditions ---\n"
        if motion_type == 'mrf':
            script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid no yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm} no no no no no no\n"
        else:
            script += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid yes yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm} no no no no no no\n"

        script += "\n; --- 4. Reports Definitions ---\n"
        script += f"/solve/report-definitions/add mflow-inlet surface-massflow surface-names {zone_inlet} () quit\n"
        script += f"/solve/report-definitions/add mflow-outlet surface-massflow surface-names {zone_outlet} () quit\n"
        
        script += "\n; --- 5. Operating & Convergence ---\n"
        script += "/define/operating-conditions/operating-pressure 0\n"
        script += f"/solve/monitors/residual/convergence-criteria {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold} {residual_threshold}\n"

        script += "\n; --- 6. Initialization ---\n"
        script += "/solve/initialize/hyb-initialization\n"

        if motion_type == 'mrf':
            script += "\n; --- 7. Solver Run (MRF) ---\n"
            script += f"/solve/iterate {max_iterations}\n"
        else:
            script_out = "\n; --- 7. Solver Run (Transient) ---\n"
            dt_1deg = (1.0 / (6.0 * rpm)) if rpm > 0 else 0
            script += f"/solve/set/time-step {dt_1deg:.6f}\n"
            script += f"/solve/set/max-iterations-per-time-step {max_iter_per_step}\n"
            script += f"/solve/dual-time-iterate {num_time_steps} {max_iter_per_step}\n"

        script += "; End Setup\n"
        
        st.text_area("Copiar no prompt do Fluent", value=script, height=350)
