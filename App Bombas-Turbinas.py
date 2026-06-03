# [ignoring loop detection]
import streamlit as st
import math

st.set_page_config(page_title="Turbomachinery Solver & Fluent TUI", layout="wide")

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
        v['W_in'] = math.sqrt(v.get('Cm1', 0)**2 + v.get('Wu1', v.get('U1', 0)-v.get('Cu1', 0))**2)
        v['W_out'] = math.sqrt(v.get('Cm2', 0)**2 + v.get('Wu2', v.get('U2', 0)-v.get('Cu2', 0))**2)

    return v


st.title("Turbomáquinas & Setup Fluent")

tab1, tab2 = st.tabs(["Cinemática & Parâmetros Base", "Scripts TUI (Fluent)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        maq_type = st.selectbox("Tipo de Máquina", ["Bomba Centrífuga", "Turbina Hidráulica"])
    with col2:
        rho = st.number_input("Densidade (kg/m³)", value=998.0)
        mu = st.number_input("Viscosidade (kg/(ms))", value=0.001003, format="%.6f")
        mod_turb = st.selectbox("Modelo de Turbulência Alvo", ["SST", "Realizable"])

    st.subheader("Entradas")

    def render_input(label, key_name):
        val_str = st.text_input(label, key=key_name)
        try:
            if val_str.strip() != "":
                return float(val_str)
        except:
            pass
        return None

    c1, c2, c3, c4 = st.columns(4)
    inputs = {}
    with c1:
        inputs['Q'] = render_input("Vazão (Q)", "in_Q")
        inputs['N'] = render_input("Rotação (RPM)", "in_N")
        inputs['D1'] = render_input("D1", "in_D1")
        inputs['b1'] = render_input("b1", "in_b1")
    with c2:
        inputs['beta1'] = render_input("beta 1", "in_beta1")
        inputs['alpha1'] = render_input("alpha 1", "in_alpha1")
        inputs['D2'] = render_input("D2", "in_D2")
        inputs['b2'] = render_input("b2", "in_b2")
    with c3:
        inputs['beta2'] = render_input("beta 2", "in_beta2")
        inputs['alpha2'] = render_input("alpha 2", "in_alpha2")
        inputs['U1'] = render_input("U1", "in_U1")
        inputs['Cu1'] = render_input("Cu1", "in_Cu1")
    with c4:
        inputs['Cm1'] = render_input("Cm1", "in_Cm1")
        inputs['U2'] = render_input("U2", "in_U2")
        inputs['Cu2'] = render_input("Cu2", "in_Cu2")
        inputs['Cm2'] = render_input("Cm2", "in_Cm2")

    inputs = {k: v for k, v in inputs.items() if v is not None}

    res = {}
    if st.button("Calcular Cinematica"):
        res = solve_kinematic_system(maq_type, rho, inputs)
        st.session_state['res'] = res
        st.subheader("Resultados")
        st.json(res)
        
        if res.get('is_complete'):
            st.success("Cálculo fechado com sucesso!")
            st.write(f"Trabalho Específico (H): {res.get('H_teo', 0):.2f} m")
            st.write(f"Vazão (Q): {res.get('Q', 0):.4f} m³/s")
            st.write(f"Potência: {res.get('Potencia', 0):.2f} W")
        else:
            st.warning("Variáveis insuficientes para fechar o triângulo de velocidades.")

with tab2:
    st.subheader("Configuração do Setup CFD (Text-User-Interface)")
    st.write("Preencha os nomes das zonas conforme criados na sua malha. Isso irá gerar um script para colar no console do Ansys Fluent para setar todos os parâmetros rapidamente.")

    motion_type = st.radio("Método", ["mrf", "mesh_motion"], format_func=lambda x: "Frame Motion (MRF - Steady)" if x == "mrf" else "Mesh Motion (Transient)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Zonas (Boundaries / Cell Zones)")
        zone_inlet = st.text_input("Inlet (Entrada)", "inlet")
        zone_outlet = st.text_input("Outlet (Saída)", "outlet")
        zone_rotor = st.text_input("Parede do Rotor", "rotor")
        zone_exterior = st.text_input("Domínio Ext. (Stator)", "interior_stator")
        zone_interior = st.text_input("Domínio Int. (Rotor Fluid)", "interior_rotor")

    with col2:
        st.write("Eixo de Rotação e CG")
        col_cg, col_ax = st.columns(2)
        with col_cg:
            cg_x = st.number_input("CG X", 0.0)
            cg_y = st.number_input("CG Y", 0.0)
            cg_z = st.number_input("CG Z", 0.0)
        with col_ax:
            ax_x = st.number_input("Eixo X", 0.0)
            ax_y = st.number_input("Eixo Y", 0.0)
            ax_z = st.number_input("Eixo Z", 1.0)
            
    if motion_type == "mrf":
        max_iterations = st.number_input("Número Máximo de Iterações", value=300)
    else:
        num_time_steps = st.number_input("Passos de Tempo (Time-Steps)", value=360)
        max_iter_per_step = st.number_input("Iterações p/ Time-Step", value=20)
        courant = st.number_input("Número de Courant (Co)", value=1.0)
        min_mesh = st.number_input("Tamanho da Menor Malha (dx em metros)", value=0.005, format="%.4f")
    
    st.write("Critérios de Convergência")
    residual_threshold = st.text_input("Resíduo Permitido", "1e-4")
    c1, c2, c3 = st.columns(3)
    monitor_continuity = c1.checkbox("Continuidade", value=True)
    monitor_velocity = c2.checkbox("Velocidades (x, y, z)", value=True)
    monitor_turbulence = c3.checkbox("Turbulência (k, ω/ε)", value=True)

    if st.button("Gerar Script TUI"):
        res = st.session_state.get('res', {})
        rpm = res.get('N', inputs.get('N', 0))
        d2 = res.get('D2', inputs.get('D2', 0))
        w_in = res.get('W_in', 0)
        w_out = res.get('W_out', 0)
        w_med = (w_in + w_out) / 2.0 if w_in > 0 and w_out > 0 else 0
        
        dt_courant = (courant * min_mesh) / w_med if w_med > 0 else 0
        dt_1deg = (1.0 / (6.0 * rpm)) if rpm > 0 else 0
        
        script_out = ""
        script_out += f"; ====== Ansys Fluent TUI Setup Script ======\n"
        script_out += f"; Generated automatically based on Turbomachinery parameters\n"
        script_out += f"; Machine: {maq_type} | RPM: {rpm:.1f}\n"
        script_out += f"; Fluid Density: {rho} | Viscosity: {mu}\n\n"

        script_out += "; --- 1. General Settings ---\n"
        script_out += "/define/models/solver/density-based-implicit no\n"
        if motion_type == 'mrf':
            script_out += "/define/models/steady yes\n"
        else:
            script_out += "/define/models/unsteady-2nd-order yes\n"

        script_out += "\n; --- 2. Turbulence Model ---\n"
        if mod_turb == 'SST':
            script_out += "/define/models/viscous/kw-sst yes\n"
        else:
            script_out += "/define/models/viscous/ke-realizable yes\n"
            
        script_out += "\n; --- 3. Materials Setup ---\n"
        script_out += f"/define/materials/change-create fluid working_fluid yes constant {rho} no no yes constant {mu} no no no\n"

        script_out += "\n; --- 4. Boundary Conditions & Zones ---\n"
        script_out += f"/define/boundary-conditions/fluid {zone_exterior} yes working_fluid no no no no 0 no 0 no 0 no 0 no 0 no 1 no no no no\n"

        if motion_type == 'mrf':
            script_out += "\n; Setup MRF (Frame Motion) on Rotor interior\n"
            script_out += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid no yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm} no no no no no no\n"
        else:
            script_out += "\n; Setup Sliding Mesh (Mesh Motion) on Rotor interior\n"
            script_out += f"/define/boundary-conditions/fluid {zone_interior} yes working_fluid yes yes {cg_x} {cg_y} {cg_z} {ax_x} {ax_y} {ax_z} {rpm} no no no no no no\n"

        script_out += "\n; --- 5. Reports Definitions ---\n"
        script_out += f"/solve/report-definitions/add mflow-inlet surface-massflow surface-names {zone_inlet} () quit\n"
        script_out += f"/solve/report-definitions/add mflow-outlet surface-massflow surface-names {zone_outlet} () quit\n"
        script_out += f"/solve/report-definitions/add torque-rotor surface-moment moment-center {cg_x} {cg_y} {cg_z} moment-axis {ax_x} {ax_y} {ax_z} surface-names {zone_rotor} () quit\n\n"
        
        freq_str = 'print yes frequency-of iteration frequency 1 quit' if motion_type == 'mrf' else 'print yes frequency-of time-step frequency 1 quit'
        
        script_out += f'/solve/report-files/add mflow-inlet-file report-defs mflow-inlet () file-name "mflow-inlet.out" ' + freq_str + '\n'
        script_out += f'/solve/report-files/add mflow-outlet-file report-defs mflow-outlet () file-name "mflow-outlet.out" ' + freq_str + '\n'
        script_out += f'/solve/report-files/add torque-rotor-file report-defs torque-rotor () file-name "torque-rotor.out" ' + freq_str + '\n'
        print(script_out)

        script_out += "\n; --- 6. Operating Conditions ---\n"
        script_out += "/define/operating-conditions/operating-pressure 0\n"

        script_out += "\n; --- 7. Convergence Criteria ---\n"
        r1 = residual_threshold if monitor_continuity else 0
        r2 = residual_threshold if monitor_velocity else 0
        r3 = residual_threshold if monitor_turbulence else 0
        script_out += f"/solve/monitors/residual/convergence-criteria {r1} {r2} {r2} {r2} {r3} {r3}\n"

        script_out += "\n; --- 8. Initialization ---\n"
        script_out += "/solve/initialize/hyb-initialization\n"

        if motion_type == 'mrf':
            script_out += "\n; --- 9. Solver Setup (MRF) ---\n"
            script_out += f"/solve/iterate {max_iterations}\n"
        else:
            script_out += "\n; --- 9. Time Step (Transient) Setup ---\n"
            script_out += f"; Using 1 degree per time-step (recommended) => dt = {dt_1deg:.6f} s\n"
            script_out += f"; Using Courant={courant:.1f} with dx={min_mesh}m => dt = {dt_courant:.6f} s\n"
            script_out += f"/solve/set/time-step {dt_1deg:.6f}\n"
            script_out += f"/solve/set/max-iterations-per-time-step {max_iter_per_step}\n"
            script_out += "\n; --- 10. Solver Setup (Transient) ---\n"
            script_out += f"/solve/dual-time-iterate {num_time_steps} {max_iter_per_step}\n"

        script_out += "\n; Script ends"
        st.code(script_out, language="text")
