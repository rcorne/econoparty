import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FPP y Restricción Presupuestaria", layout="wide")

st.title("FPP y Restricción Presupuestaria")
st.markdown("Haz clic en cualquier punto del gráfico para analizar qué ocurre ahí.")

# ─── Parámetros ───────────────────────────────────────────────────────────────
A = 100.0
B = 100.0
I = 100.0
PX = 1.0
PY = 1.0

# ─── FPP: cuarto de elipse ────────────────────────────────────────────────────
x_fpp = np.linspace(0, A, 400)
y_fpp = B * np.sqrt(np.maximum(1 - (x_fpp / A)**2, 0))

# ─── Restricción Presupuestaria ───────────────────────────────────────────────
x_rp = np.linspace(0, I / PX, 400)
y_rp = (I - PX * x_rp) / PY

# ─── Óptimos ─────────────────────────────────────────────────────────────────
x_opt_soc  = A / np.sqrt(2)
y_opt_soc  = B / np.sqrt(2)
x_opt_cons = I / (2 * PX)
y_opt_cons = I / (2 * PY)
U_opt      = np.sqrt(x_opt_cons * y_opt_cons)

# ─── Curvas de indiferencia consumidor ───────────────────────────────────────
x_ci = np.linspace(1, 105, 400)

def ci_y(U, x):
    with np.errstate(divide="ignore", invalid="ignore"):
        y = U**2 / np.maximum(x, 1e-6)
    return np.where(y > 110, np.nan, y)

# ─── CI social ───────────────────────────────────────────────────────────────
x_ci_soc = np.linspace(1, 99, 400)
U_soc     = np.sqrt(x_opt_soc * y_opt_soc)
y_ci_soc  = np.where(x_ci_soc > 0, U_soc**2 / x_ci_soc, np.nan)
y_ci_soc  = np.where(y_ci_soc > 105, np.nan, y_ci_soc)

# ─── Clasificadores ───────────────────────────────────────────────────────────
def clasificar_fpp(px, py):
    if px < 0 or py < 0 or px > A + 5 or py > B + 5:
        return "fuera"
    y_front  = B * np.sqrt(max(1 - (px / A)**2, 0))
    dist_opt = np.sqrt((px - x_opt_soc)**2 + (py - y_opt_soc)**2)
    if py > y_front + 2:
        return "inalcanzable"
    elif abs(py - y_front) <= 2:
        return "optimo_social" if dist_opt < 10 else "eficiente_no_optimo"
    else:
        return "ineficiente"

def clasificar_rp(px, py):
    if px < 0 or py < 0:
        return "fuera"
    gasto    = PX * px + PY * py
    dist_opt = np.sqrt((px - x_opt_cons)**2 + (py - y_opt_cons)**2)
    if gasto > I + 2:
        return "inalcanzable"
    elif abs(gasto - I) <= 2:
        return "optimo_consumidor" if dist_opt < 8 else "sobre_rp_no_optimo"
    else:
        return "interior_rp"

# ─── Textos ───────────────────────────────────────────────────────────────────
INFO_FPP = {
    "inalcanzable": {
        "icono": "🚫", "color": "#E24B4A",
        "titulo": "Punto inalcanzable",
        "texto": (
            "Este punto está **fuera de la FPP**. La economía no puede producir "
            "esta combinación con los recursos y tecnología disponibles.\n\n"
            "Para alcanzarlo se necesitaría más tecnología, más recursos, o ambos. "
            "En el corto plazo es simplemente imposible.\n\n"
            "**Concepto clave:** La FPP delimita lo que es técnicamente *factible*."
        )
    },
    "ineficiente": {
        "icono": "⚠️", "color": "#EF9F27",
        "titulo": "Interior — ineficiencia productiva",
        "texto": (
            "Este punto está **dentro de la FPP**. La economía no usa todos sus "
            "recursos eficientemente: hay desempleo, capacidad ociosa o ineficiencia "
            "organizacional.\n\n"
            "Desde aquí es posible producir **más de ambos bienes** sin sacrificar nada. "
            "Moverse hacia la FPP es una mejora de Pareto pura.\n\n"
            "**Eficiencia productiva:** ❌ No se cumple.\n\n"
            "**Mensaje del economista:** Llegar a la FPP es condición *necesaria* "
            "para el bienestar, pero aún no suficiente."
        )
    },
    "eficiente_no_optimo": {
        "icono": "🟡", "color": "#BA7517",
        "titulo": "Sobre la FPP — productivamente eficiente, no asignativo",
        "texto": (
            "Este punto está **sobre la FPP**: no hay recursos desperdiciados y no es "
            "posible producir más de un bien sin sacrificar el otro.\n\n"
            "Sin embargo, **no es el óptimo social**: la combinación no maximiza el "
            "bienestar colectivo. Para saber cuál es el punto correcto se necesitan "
            "las *preferencias de la sociedad*.\n\n"
            "**Eficiencia productiva:** ✅\n\n"
            "**Eficiencia asignativa:** ❌\n\n"
            "**Condición faltante:** TMT = TMS social (no se cumple aquí)."
        )
    },
    "optimo_social": {
        "icono": "✅", "color": "#1D9E75",
        "titulo": "E* — Óptimo social",
        "texto": (
            "Este es el **óptimo social**: la economía produce sobre la FPP *y* en "
            "el punto que maximiza el bienestar colectivo.\n\n"
            "La curva de indiferencia social más alta posible es **tangente** a la "
            "FPP aquí. La Tasa Marginal de Transformación (TMT) iguala la Tasa "
            "Marginal de Sustitución social (TMS).\n\n"
            "**Eficiencia productiva:** ✅\n\n"
            "**Eficiencia asignativa:** ✅\n\n"
            "**Condición de óptimo:** TMT = TMS social\n\n"
            "Para identificar este punto se necesitaron las **preferencias de la "
            "sociedad** — no basta con saber la tecnología disponible."
        )
    },
    "fuera": {
        "icono": "↩️", "color": "#888780",
        "titulo": "Fuera del área",
        "texto": "Haz clic dentro del área del gráfico para analizar un punto."
    }
}

INFO_RP = {
    "inalcanzable": {
        "icono": "🚫", "color": "#E24B4A",
        "titulo": "Punto inalcanzable",
        "texto": (
            "Este punto está **fuera de la restricción presupuestaria**. El "
            "consumidor no puede comprar esta combinación con su ingreso actual.\n\n"
            "El gasto requerido supera el ingreso disponible (I = 100).\n\n"
            "**Concepto clave:** La RP delimita lo que es *financieramente* factible."
        )
    },
    "interior_rp": {
        "icono": "💤", "color": "#888780",
        "titulo": "Interior — ingreso no agotado",
        "texto": (
            "Este punto está **dentro de la restricción presupuestaria**. El "
            "consumidor no está gastando todo su ingreso.\n\n"
            "Con el supuesto de **no saciedad**, el consumidor siempre puede "
            "mejorar su bienestar gastando más. Este punto nunca es óptimo.\n\n"
            "**Condición de óptimo:** ❌ Hay mejoras posibles sin violar el presupuesto."
        )
    },
    "sobre_rp_no_optimo": {
        "icono": "🟡", "color": "#BA7517",
        "titulo": "Sobre la RP — agota ingreso, pero no es óptimo",
        "texto": (
            "Este punto **agota el ingreso** (Px·x + Py·y = I ✓), pero no es "
            "el óptimo del consumidor.\n\n"
            "La curva de indiferencia **cruza** la restricción aquí, lo que "
            "significa que existe otro punto sobre la RP con mayor bienestar.\n\n"
            "**Condición faltante:** TMS = Px/Py no se cumple en este punto.\n\n"
            "La pendiente de la curva de indiferencia no iguala la pendiente "
            "de la RP: el consumidor puede reordenar su gasto y estar mejor."
        )
    },
    "optimo_consumidor": {
        "icono": "✅", "color": "#1D9E75",
        "titulo": "E* — Óptimo del consumidor",
        "texto": (
            "Este es el **óptimo del consumidor**: maximiza su bienestar dado "
            "su ingreso y los precios.\n\n"
            "La curva de indiferencia es **tangente** a la RP — no la cruza.\n\n"
            "**Condición de óptimo:** TMS = Px/Py ✅\n\n"
            "La valoración subjetiva del bien X respecto al Y iguala exactamente "
            "su precio relativo de mercado. No hay reorganización posible del gasto "
            "que mejore el bienestar.\n\n"
            "Esto conecta con la **pregunta 5**: el consumidor elige hasta que su "
            "valoración marginal iguala el precio de mercado."
        )
    },
    "fuera": {
        "icono": "↩️", "color": "#888780",
        "titulo": "Fuera del área",
        "texto": "Haz clic dentro del área del gráfico para analizar un punto."
    }
}

# ─── Builders de figuras Plotly ───────────────────────────────────────────────
def build_fpp_fig(punto=None):
    fig = go.Figure()

    # Rellenos de zonas
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fpp, x_fpp[::-1]]),
        y=np.concatenate([y_fpp, np.full_like(y_fpp, 108)]),
        fill="toself", fillcolor="rgba(226,75,74,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0], x_fpp, [A, 0]]),
        y=np.concatenate([[0], y_fpp, [0, 0]]),
        fill="toself", fillcolor="rgba(24,95,165,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))

    # FPP
    fig.add_trace(go.Scatter(
        x=x_fpp, y=y_fpp, mode="lines",
        line=dict(color="#185FA5", width=3), name="FPP",
        hovertemplate="FPP: (%{x:.1f}, %{y:.1f})<extra></extra>"
    ))

    # CI social
    mask = ~np.isnan(y_ci_soc)
    fig.add_trace(go.Scatter(
        x=x_ci_soc[mask], y=y_ci_soc[mask], mode="lines",
        line=dict(color="#1D9E75", width=1.5, dash="dash"),
        name="CI social óptima", hoverinfo="skip"
    ))

    # Óptimo social
    fig.add_trace(go.Scatter(
        x=[x_opt_soc], y=[y_opt_soc], mode="markers+text",
        marker=dict(color="#1D9E75", size=12, line=dict(color="white", width=2)),
        text=["E*"], textposition="top right",
        textfont=dict(color="#0F6E56", size=13),
        name="E* social",
        hovertemplate=f"E* social: ({x_opt_soc:.1f}, {y_opt_soc:.1f})<extra></extra>"
    ))

    # Anotaciones de zonas
    fig.add_annotation(x=78, y=97, text="Inalcanzable",
                       showarrow=False, font=dict(color="#E24B4A", size=11), opacity=0.7)
    fig.add_annotation(x=10, y=12, text="Ineficiente (interior)",
                       showarrow=False, font=dict(color="#185FA5", size=11), opacity=0.7)

    # Punto clickeado
    if punto:
        px, py = punto
        zona  = clasificar_fpp(px, py)
        color = INFO_FPP[zona]["color"]
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers",
            marker=dict(color=color, size=15, symbol="x-thin",
                        line=dict(color=color, width=3)),
            showlegend=False,
            hovertemplate=f"Seleccionado: ({px:.1f}, {py:.1f})<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="Bien X", range=[0, 108], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="Bien Y", range=[0, 108], gridcolor="#eee", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(x=0.55, y=0.97, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd", borderwidth=1),
        height=420, clickmode="event"
    )
    return fig

def build_rp_fig(punto=None):
    fig = go.Figure()

    # Rellenos de zonas
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_rp, x_rp[::-1]]),
        y=np.concatenate([y_rp, np.full_like(y_rp, 108)]),
        fill="toself", fillcolor="rgba(226,75,74,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0], x_rp, [I/PX, 0]]),
        y=np.concatenate([[0], y_rp, [0, 0]]),
        fill="toself", fillcolor="rgba(83,74,183,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))

    # RP
    fig.add_trace(go.Scatter(
        x=x_rp, y=y_rp, mode="lines",
        line=dict(color="#534AB7", width=3),
        name="RP (Px·x + Py·y = I)",
        hovertemplate="RP: (%{x:.1f}, %{y:.1f})<extra></extra>"
    ))

    # Curvas de indiferencia
    for U_val, dash, width, name in [
        (U_opt * 0.65, "dash",  1.2, "U₁"),
        (U_opt,        "solid", 2.0, "U₂ (óptimo)"),
        (U_opt * 1.30, "dash",  1.2, "U₃"),
    ]:
        y_val = ci_y(U_val, x_ci)
        mask  = ~np.isnan(y_val) & (y_val >= 0)
        fig.add_trace(go.Scatter(
            x=x_ci[mask], y=y_val[mask], mode="lines",
            line=dict(color="#1D9E75", width=width, dash=dash),
            name=name, hoverinfo="skip"
        ))

    # Óptimo consumidor
    fig.add_trace(go.Scatter(
        x=[x_opt_cons], y=[y_opt_cons], mode="markers+text",
        marker=dict(color="#1D9E75", size=12, line=dict(color="white", width=2)),
        text=["E*"], textposition="top right",
        textfont=dict(color="#0F6E56", size=13),
        name="E* consumidor",
        hovertemplate=f"E*: ({x_opt_cons:.1f}, {y_opt_cons:.1f})<extra></extra>"
    ))

    # Anotaciones de zonas
    fig.add_annotation(x=75, y=97, text="Inalcanzable",
                       showarrow=False, font=dict(color="#E24B4A", size=11), opacity=0.7)
    fig.add_annotation(x=8, y=10, text="Ingreso no gastado",
                       showarrow=False, font=dict(color="#534AB7", size=11), opacity=0.7)

    # Punto clickeado
    if punto:
        px, py = punto
        zona  = clasificar_rp(px, py)
        color = INFO_RP[zona]["color"]
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers",
            marker=dict(color=color, size=15, symbol="x-thin",
                        line=dict(color=color, width=3)),
            showlegend=False,
            hovertemplate=f"Seleccionado: ({px:.1f}, {py:.1f})<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="Bien X", range=[0, 108], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="Bien Y", range=[0, 108], gridcolor="#eee", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(x=0.55, y=0.97, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd", borderwidth=1),
        height=420, clickmode="event"
    )
    return fig

# ─── Estado de sesión ─────────────────────────────────────────────────────────
if "punto_fpp" not in st.session_state:
    st.session_state.punto_fpp = None
if "punto_rp" not in st.session_state:
    st.session_state.punto_rp = None

# ─── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Frontera de Posibilidades de Producción")
    ev_fpp = st.plotly_chart(
        build_fpp_fig(st.session_state.punto_fpp),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="chart_fpp"
    )
    if ev_fpp and ev_fpp.selection and ev_fpp.selection.points:
        pt = ev_fpp.selection.points[0]
        st.session_state.punto_fpp = (pt["x"], pt["y"])

    if st.session_state.punto_fpp:
        px, py = st.session_state.punto_fpp
        zona  = clasificar_fpp(px, py)
        info  = INFO_FPP[zona]
        color = info["color"]
        st.markdown(f"""
<div style="background:{color}18;border-left:4px solid {color};
     padding:12px 16px;border-radius:6px;margin-top:4px;">
<span style="font-size:1.2em">{info['icono']}</span>
<strong style="color:{color}"> {info['titulo']}</strong><br>
<small style="color:#666">Punto: ({px:.1f}, {py:.1f})</small>
</div>""", unsafe_allow_html=True)
        st.markdown(info["texto"])
    else:
        st.info("Haz clic en el gráfico para analizar un punto.")

with col2:
    st.subheader("Restricción Presupuestaria")
    ev_rp = st.plotly_chart(
        build_rp_fig(st.session_state.punto_rp),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="chart_rp"
    )
    if ev_rp and ev_rp.selection and ev_rp.selection.points:
        pt = ev_rp.selection.points[0]
        st.session_state.punto_rp = (pt["x"], pt["y"])

    if st.session_state.punto_rp:
        px, py = st.session_state.punto_rp
        zona  = clasificar_rp(px, py)
        info  = INFO_RP[zona]
        color = info["color"]
        gasto = PX * px + PY * py
        st.markdown(f"""
<div style="background:{color}18;border-left:4px solid {color};
     padding:12px 16px;border-radius:6px;margin-top:4px;">
<span style="font-size:1.2em">{info['icono']}</span>
<strong style="color:{color}"> {info['titulo']}</strong><br>
<small style="color:#666">Punto: ({px:.1f}, {py:.1f}) — Gasto: {gasto:.1f} / {I:.0f}</small>
</div>""", unsafe_allow_html=True)
        st.markdown(info["texto"])
    else:
        st.info("Haz clic en el gráfico para analizar un punto.")

# ─── Tabla resumen ────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Guía de zonas")
st.markdown("""
| Zona | FPP | Restricción Presupuestaria |
|---|---|---|
| 🚫 Inalcanzable | Requiere más tecnología o recursos | Requiere más ingreso |
| ⚠️ Interior | Ineficiencia productiva — recursos ociosos | Ingreso no gastado |
| 🟡 Frontera incorrecta | Ef. productiva ✅, asignativa ❌ | Gasta todo, pero TMS ≠ Px/Py |
| ✅ Óptimo | Ef. productiva ✅ + asignativa ✅ | Máximo bienestar dado ingreso |
""")
st.caption("Parámetros: I = 100, Px = Py = 1. FPP modelada como cuarto de elipse (semiejes = 100).")
