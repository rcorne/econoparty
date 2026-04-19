import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

st.set_page_config(page_title="FPP y Restricción Presupuestaria", layout="wide")

st.title("FPP y Restricción Presupuestaria — Explorador Interactivo")
st.markdown("Haz clic en cualquier punto de cada gráfico para entender qué ocurre en ese punto.")

# ─── Parámetros ───────────────────────────────────────────────────────────────
T_MAX   = 100   # máximo bien X (o trigo, o bien en eje horizontal)
Y_MAX   = 100   # máximo bien Y
INGRESO = 100   # ingreso del consumidor
PX      = 1.0   # precio bien X
PY      = 1.0   # precio bien Y

# ─── FPP: cuarto de elipse  x²/a² + y²/b² = 1 ────────────────────────────────
a_fpp = T_MAX
b_fpp = Y_MAX
x_fpp = np.linspace(0, a_fpp, 400)
y_fpp = b_fpp * np.sqrt(np.maximum(1 - (x_fpp / a_fpp)**2, 0))

# ─── Restricción Presupuestaria: Px·x + Py·y = I ─────────────────────────────
x_rp = np.linspace(0, INGRESO / PX, 400)
y_rp = (INGRESO - PX * x_rp) / PY

# ─── Curvas de indiferencia del consumidor (U = x^0.5 * y^0.5) ───────────────
def curva_indiferencia(U, x_vals):
    with np.errstate(divide='ignore', invalid='ignore'):
        y = U**2 / np.maximum(x_vals, 1e-6)
    return y

# Óptimo consumidor: x* = I/(2Px), y* = I/(2Py)
x_opt_cons = INGRESO / (2 * PX)
y_opt_cons = INGRESO / (2 * PY)
U_opt = np.sqrt(x_opt_cons * y_opt_cons)

x_ci = np.linspace(2, INGRESO / PX, 400)
U_bajo  = U_opt * 0.65
U_alto  = U_opt * 1.25

# ─── Óptimo social en la FPP (punto de tangencia con CI social) ──────────────
# Con utilidad social U=x^0.5*y^0.5 y FPP elíptica, el óptimo es (a/√2, b/√2)
x_opt_soc = a_fpp / np.sqrt(2)
y_opt_soc = b_fpp / np.sqrt(2)

# Curva de indiferencia social tangente en E*
x_ci_soc = np.linspace(2, a_fpp - 1, 400)
U_soc_opt = np.sqrt(x_opt_soc * y_opt_soc)
y_ci_soc  = U_soc_opt**2 / np.maximum(x_ci_soc, 1e-6)
mask_soc  = y_ci_soc <= Y_MAX + 5

# ─── Función para clasificar punto en la FPP ─────────────────────────────────
def clasificar_fpp(px, py):
    """Devuelve (zona, y_frontera, descripcion)"""
    if px < 0 or py < 0 or px > a_fpp or py > b_fpp:
        return "fuera", None, None
    y_frontera = b_fpp * np.sqrt(max(1 - (px / a_fpp)**2, 0))
    dist_opt   = np.sqrt((px - x_opt_soc)**2 + (py - y_opt_soc)**2)

    if py > y_frontera + 1.5:
        zona = "inalcanzable"
    elif abs(py - y_frontera) <= 1.5:
        if dist_opt < 8:
            zona = "optimo_social"
        else:
            zona = "eficiente_no_optimo"
    else:
        zona = "ineficiente"
    return zona, y_frontera, dist_opt

def clasificar_rp(px, py):
    """Devuelve zona respecto a la RP"""
    if px < 0 or py < 0 or px > INGRESO/PX + 2 or py > INGRESO/PY + 2:
        return "fuera", None
    gasto = PX * px + PY * py
    U_punto = np.sqrt(max(px, 0) * max(py, 0))
    dist_opt = np.sqrt((px - x_opt_cons)**2 + (py - y_opt_cons)**2)

    if gasto > INGRESO + 1.5:
        zona = "inalcanzable"
    elif abs(gasto - INGRESO) <= 1.5:
        if dist_opt < 6:
            zona = "optimo_consumidor"
        else:
            zona = "sobre_rp_no_optimo"
    else:
        zona = "interior_rp"
    return zona, gasto

# ─── Textos explicativos ──────────────────────────────────────────────────────
textos_fpp = {
    "inalcanzable": {
        "titulo": "Punto inalcanzable",
        "color": "#E24B4A",
        "icono": "🚫",
        "texto": """
**Este punto está fuera de la FPP.** La economía no puede producir esta combinación con los recursos y tecnología disponibles.

- La FPP es el límite máximo de producción posible.
- Para alcanzar este punto se necesitaría más tecnología, más recursos, o ambos.
- En el corto plazo, es simplemente imposible.

**Concepto clave:** La FPP delimita lo que es técnicamente *factible*.
        """
    },
    "ineficiente": {
        "titulo": "Punto interior — ineficiencia productiva",
        "color": "#E24B4A",
        "icono": "⚠️",
        "texto": """
**Este punto está dentro de la FPP.** La economía no está usando todos sus recursos eficientemente.

- Existen recursos ociosos: desempleo, capacidad instalada sin usar, ineficiencia organizacional.
- Es posible producir **más de ambos bienes** sin sacrificar nada.
- Moverse hacia la FPP es una mejora de Pareto pura.

**Eficiencia productiva:** ❌ No se cumple.
**Eficiencia asignativa:** ❌ No aplica todavía.

**Mensaje del economista:** Primero hay que llegar a la FPP — eso es condición *necesaria*.
        """
    },
    "eficiente_no_optimo": {
        "titulo": "Sobre la FPP — eficiencia productiva, no asignativa",
        "color": "#EF9F27",
        "icono": "🟡",
        "texto": """
**Este punto está sobre la FPP.** La economía es productivamente eficiente: no hay recursos desperdiciados.

- No es posible producir más de un bien sin sacrificar el otro.
- Sin embargo, este **no es el óptimo social**: la combinación no maximiza el bienestar colectivo.
- Para saber si es el punto correcto, se necesitan las *preferencias sociales*.

**Eficiencia productiva:** ✅ Se cumple.
**Eficiencia asignativa:** ❌ No se cumple.

**Mensaje del economista:** Llegar a la FPP es condición necesaria pero **no suficiente**. Hay que elegir el punto *correcto* sobre ella.
        """
    },
    "optimo_social": {
        "titulo": "E* — Óptimo social (eficiencia productiva + asignativa)",
        "color": "#1D9E75",
        "icono": "✅",
        "texto": """
**Este es el óptimo social.** La economía produce sobre la FPP *y* en el punto que maximiza el bienestar colectivo.

- La tasa marginal de transformación (TMT) iguala la tasa marginal de sustitución social (TMS).
- La curva de indiferencia social más alta posible es **tangente** a la FPP aquí.
- Para identificar este punto se necesitaron las **preferencias de la sociedad**.

**Eficiencia productiva:** ✅ Se cumple (sobre la FPP).
**Eficiencia asignativa:** ✅ Se cumple (punto correcto sobre la FPP).

**Condición de óptimo:** TMT = TMS social
        """
    },
}

textos_rp = {
    "inalcanzable": {
        "titulo": "Punto inalcanzable",
        "color": "#E24B4A",
        "icono": "🚫",
        "texto": """
**Este punto está fuera de la restricción presupuestaria.** El consumidor no puede comprar esta combinación con su ingreso actual.

- El gasto requerido supera el ingreso disponible.
- Para alcanzarlo, el consumidor necesitaría más ingreso o precios más bajos.

**Concepto clave:** La RP delimita lo que es *financieramente* factible para el individuo.
        """
    },
    "interior_rp": {
        "titulo": "Punto interior — ingreso no agotado",
        "color": "#888780",
        "icono": "💤",
        "texto": """
**Este punto está dentro de la restricción presupuestaria.** El consumidor no está gastando todo su ingreso.

- Queda ingreso sin usar: el consumidor podría comprar más de algún bien y aumentar su bienestar.
- Con el supuesto de **no saciedad**, el consumidor siempre prefiere gastar todo su ingreso.
- Este punto nunca es óptimo bajo preferencias bien comportadas.

**Condición de óptimo:** ❌ No se cumple — hay mejoras posibles sin violar el presupuesto.
        """
    },
    "sobre_rp_no_optimo": {
        "titulo": "Sobre la RP — agota el ingreso, pero no es óptimo",
        "color": "#EF9F27",
        "icono": "🟡",
        "texto": """
**Este punto agota el ingreso** (está sobre la RP), pero no es el óptimo del consumidor.

- El gasto total es igual al ingreso: Px·x + Py·y = I ✓
- Sin embargo, la curva de indiferencia **cruza** la restricción en este punto.
- Existe otro punto sobre la RP que pertenece a una curva de indiferencia **más alta**.
- La TMS ≠ razón de precios (Px/Py): las pendientes no son iguales.

**Condición de óptimo:** ❌ No se cumple — TMS ≠ Px/Py.
        """
    },
    "optimo_consumidor": {
        "titulo": "E* — Óptimo del consumidor",
        "color": "#1D9E75",
        "icono": "✅",
        "texto": """
**Este es el óptimo del consumidor.** Maximiza su bienestar dado su ingreso y los precios.

- El gasto iguala exactamente el ingreso: Px·x + Py·y = I ✓
- La curva de indiferencia es **tangente** a la RP: no la cruza.
- La TMS = razón de precios (Px/Py): la valoración subjetiva iguala el precio de mercado.

**Condición de óptimo:** ✅ TMS = Px/Py

Esto conecta con la pregunta 5: el consumidor trabaja/consume hasta que su valoración subjetiva de un bien iguala su precio relativo de mercado.
        """
    },
}

# ─── Estado de sesión ─────────────────────────────────────────────────────────
if "click_fpp" not in st.session_state:
    st.session_state.click_fpp = None
if "click_rp" not in st.session_state:
    st.session_state.click_rp = None

# ─── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1: FPP
# ══════════════════════════════════════════════════════════════════════════════
with col1:
    st.subheader("Frontera de Posibilidades de Producción")
    st.caption("Haz clic en el gráfico para analizar ese punto")

    fig1, ax1 = plt.subplots(figsize=(5.5, 5))
    ax1.set_facecolor("#fafafa")
    fig1.patch.set_facecolor("#fafafa")

    # FPP
    ax1.plot(x_fpp, y_fpp, color="#185FA5", lw=2.5, label="FPP", zorder=3)

    # Zona inalcanzable (relleno)
    ax1.fill_between(x_fpp, y_fpp, Y_MAX + 5,
                     color=to_rgba("#E24B4A", 0.07), zorder=1)

    # Zona interior (ineficiente)
    ax1.fill_between(x_fpp, 0, y_fpp,
                     color=to_rgba("#185FA5", 0.06), zorder=1)

    # Curva de indiferencia social óptima
    mask = (y_ci_soc >= 0) & (y_ci_soc <= Y_MAX + 2) & mask_soc
    ax1.plot(x_ci_soc[mask], y_ci_soc[mask],
             color="#1D9E75", lw=1.4, ls="--", label="CI social óptima", zorder=2)

    # Punto óptimo social E*
    ax1.scatter([x_opt_soc], [y_opt_soc], color="#1D9E75", s=90, zorder=5)
    ax1.annotate("E* (óptimo social)", (x_opt_soc, y_opt_soc),
                 xytext=(x_opt_soc + 6, y_opt_soc + 4),
                 fontsize=8, color="#0F6E56",
                 arrowprops=dict(arrowstyle="->", color="#0F6E56", lw=0.8))

    # Punto clickeado
    if st.session_state.click_fpp:
        px, py = st.session_state.click_fpp
        zona, y_front, dist = clasificar_fpp(px, py)
        color_punto = textos_fpp.get(zona, {}).get("color", "#888780")
        ax1.scatter([px], [py], color=color_punto, s=110,
                    zorder=6, edgecolors="white", linewidths=1.5)
        ax1.annotate(f"({px:.0f}, {py:.0f})", (px, py),
                     xytext=(px + 3, py + 3), fontsize=7.5, color=color_punto)

    # Anotaciones de zonas
    ax1.text(75, 85, "Inalcanzable", fontsize=8,
             color="#E24B4A", alpha=0.7, style="italic")
    ax1.text(10, 15, "Ineficiente\n(interior)", fontsize=8,
             color="#185FA5", alpha=0.7, style="italic")

    ax1.set_xlim(0, T_MAX + 5)
    ax1.set_ylim(0, Y_MAX + 8)
    ax1.set_xlabel("Bien X", fontsize=11)
    ax1.set_ylabel("Bien Y", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Haz clic para analizar un punto", fontsize=9, color="#555")

    # Click handler
    clicked1 = st.pyplot(fig1, use_container_width=True)

    # Input manual de coordenadas
    st.markdown("**Ingresa coordenadas del punto (o usa los sliders):**")
    c1a, c1b = st.columns(2)
    with c1a:
        x_click1 = st.slider("Bien X", 0, 110, 50, key="x_fpp")
    with c1b:
        y_click1 = st.slider("Bien Y", 0, 110, 50, key="y_fpp")

    if st.button("Analizar este punto →", key="btn_fpp"):
        st.session_state.click_fpp = (x_click1, y_click1)
        st.rerun()

    # Explicación
    if st.session_state.click_fpp:
        px, py = st.session_state.click_fpp
        zona, y_front, dist = clasificar_fpp(px, py)
        info = textos_fpp.get(zona)
        if info:
            color = info["color"]
            st.markdown(f"""
<div style="background:{color}18; border-left: 4px solid {color};
     padding: 12px 16px; border-radius: 6px; margin-top: 8px;">
<b>{info['icono']} {info['titulo']}</b><br>
<small>Punto seleccionado: ({px:.1f}, {py:.1f})</small>
</div>
""", unsafe_allow_html=True)
            st.markdown(info["texto"])
            if y_front is not None:
                st.caption(f"Producción máxima de Bien Y dado X={px:.1f}: **{y_front:.1f} unidades**")

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2: Restricción Presupuestaria
# ══════════════════════════════════════════════════════════════════════════════
with col2:
    st.subheader("Restricción Presupuestaria")
    st.caption("Haz clic en el gráfico para analizar ese punto")

    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    ax2.set_facecolor("#fafafa")
    fig2.patch.set_facecolor("#fafafa")

    # RP
    ax2.plot(x_rp, y_rp, color="#534AB7", lw=2.5, label="RP (Px·x + Py·y = I)", zorder=3)

    # Zona inalcanzable
    ax2.fill_between(x_rp, y_rp, Y_MAX + 5,
                     color=to_rgba("#E24B4A", 0.07), zorder=1)

    # Zona interior (ingreso no gastado)
    ax2.fill_between(x_rp, 0, y_rp,
                     color=to_rgba("#534AB7", 0.06), zorder=1)

    # Curvas de indiferencia
    for U_val, ls, lw, alpha, label in [
        (U_bajo,   "--", 1.2, 0.7, "U₁ (menor bienestar)"),
        (U_opt,    "-",  1.8, 1.0, "U₂ (óptimo)"),
        (U_alto,   "--", 1.2, 0.7, "U₃ (inalcanzable)"),
    ]:
        y_ci_val = curva_indiferencia(U_val, x_ci)
        mask_ci  = (y_ci_val >= 0) & (y_ci_val <= Y_MAX + 2)
        ax2.plot(x_ci[mask_ci], y_ci_val[mask_ci],
                 color="#1D9E75", lw=lw, ls=ls, alpha=alpha,
                 label=label, zorder=2)

    # Punto óptimo consumidor
    ax2.scatter([x_opt_cons], [y_opt_cons], color="#1D9E75", s=90, zorder=5)
    ax2.annotate("E* (óptimo)", (x_opt_cons, y_opt_cons),
                 xytext=(x_opt_cons + 5, y_opt_cons + 4),
                 fontsize=8, color="#0F6E56",
                 arrowprops=dict(arrowstyle="->", color="#0F6E56", lw=0.8))

    # Punto clickeado
    if st.session_state.click_rp:
        px, py = st.session_state.click_rp
        zona, gasto = clasificar_rp(px, py)
        color_punto = textos_rp.get(zona, {}).get("color", "#888780")
        ax2.scatter([px], [py], color=color_punto, s=110,
                    zorder=6, edgecolors="white", linewidths=1.5)
        ax2.annotate(f"({px:.0f}, {py:.0f})", (px, py),
                     xytext=(px + 2, py + 3), fontsize=7.5, color=color_punto)

    # Anotaciones zonas
    ax2.text(72, 88, "Inalcanzable", fontsize=8,
             color="#E24B4A", alpha=0.7, style="italic")
    ax2.text(8, 12, "Ingreso\nno gastado", fontsize=8,
             color="#534AB7", alpha=0.7, style="italic")

    ax2.set_xlim(0, INGRESO / PX + 8)
    ax2.set_ylim(0, INGRESO / PY + 8)
    ax2.set_xlabel("Bien X", fontsize=11)
    ax2.set_ylabel("Bien Y", fontsize=11)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Haz clic para analizar un punto", fontsize=9, color="#555")

    st.pyplot(fig2, use_container_width=True)

    st.markdown("**Ingresa coordenadas del punto (o usa los sliders):**")
    c2a, c2b = st.columns(2)
    with c2a:
        x_click2 = st.slider("Bien X", 0, 110, 50, key="x_rp")
    with c2b:
        y_click2 = st.slider("Bien Y", 0, 110, 50, key="y_rp")

    if st.button("Analizar este punto →", key="btn_rp"):
        st.session_state.click_rp = (x_click2, y_click2)
        st.rerun()

    # Explicación
    if st.session_state.click_rp:
        px, py = st.session_state.click_rp
        zona, gasto = clasificar_rp(px, py)
        info = textos_rp.get(zona)
        if info:
            color = info["color"]
            st.markdown(f"""
<div style="background:{color}18; border-left: 4px solid {color};
     padding: 12px 16px; border-radius: 6px; margin-top: 8px;">
<b>{info['icono']} {info['titulo']}</b><br>
<small>Punto seleccionado: ({px:.1f}, {py:.1f}) — Gasto total: {gasto:.1f} / {INGRESO}</small>
</div>
""", unsafe_allow_html=True)
            st.markdown(info["texto"])

# ─── Pie de página ────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**Guía rápida de zonas:**

| Zona | FPP | Restricción Presupuestaria |
|---|---|---|
| 🔴 Inalcanzable | Requiere más tecnología/recursos | Requiere más ingreso |
| ⚠️ Interior | Ineficiencia productiva (recursos ociosos) | Ingreso no gastado |
| 🟡 Sobre la frontera, punto incorrecto | Ef. productiva ✅, asignativa ❌ | Gasta todo, pero TMS ≠ Px/Py |
| ✅ Óptimo | Ef. productiva ✅ + asignativa ✅ | Máximo bienestar dado ingreso |
""")

st.caption("Parámetros: Ingreso = 100, Px = Py = 1. FPP modelada como cuarto de elipse con ejes = 100.")
