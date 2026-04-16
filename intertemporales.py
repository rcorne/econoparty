"""
Simulador interactivo: Efecto Sustitución y Efecto Ingreso
Econometría / Microeconomía
Ejecutar con: streamlit run efecto_ingreso_sustitucion.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

st.set_page_config(page_title="Efecto Sustitución e Ingreso", layout="wide")

st.title("Efecto Sustitución y Efecto Ingreso")
st.markdown("Simulador interactivo con función de utilidad Cobb-Douglas: $U(x,y) = x^{0.5} \\cdot y^{0.5}$")

# ── Controles ──────────────────────────────────────────────────────────────────
col_ctrl, col_graf = st.columns([1, 2.2])

with col_ctrl:
    st.subheader("Parámetros")
    Px  = st.slider("Precio bien X  (Px)",  min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    Px2 = st.slider("Nuevo precio X (Px')", min_value=0.5, max_value=Px,  value=max(0.5, Px - 0.5), step=0.5)
    Py  = st.slider("Precio bien Y  (Py)",  min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    M   = st.slider("Ingreso        (M)",   min_value=10,  max_value=30,  value=20,  step=1)

    st.divider()
    st.subheader("Mostrar pasos")
    show_A  = st.checkbox("1. Equilibrio inicial (A)",        value=True)
    show_B  = st.checkbox("2. Nuevo equilibrio (B)",          value=False)
    show_C  = st.checkbox("3. Efecto sustitución → C",        value=False)
    show_EI = st.checkbox("4. Efecto ingreso  (C → B)",       value=False)

# ── Cálculos ───────────────────────────────────────────────────────────────────
alpha = 0.5

def optimo(M_, Px_, Py_):
    x = alpha * M_ / Px_
    y = (1 - alpha) * M_ / Py_
    return x, y

def utilidad(x, y):
    if x <= 0 or y <= 0:
        return 0.0
    return (x ** alpha) * (y ** (1 - alpha))

def curva_indiferencia(U, x_arr):
    y_arr = np.full_like(x_arr, np.nan)
    mask = x_arr > 0
    y_arr[mask] = (U / (x_arr[mask] ** alpha)) ** (1 / (1 - alpha))
    return y_arr

xA, yA = optimo(M, Px, Py)
UA     = utilidad(xA, yA)

xB, yB = optimo(M, Px2, Py)
UB     = utilidad(xB, yB)

# Punto C: minimizar gasto en la curva UA con precios nuevos
# Con Cobb-Douglas: x_C = α·UA / [α^α·(1-α)^(1-α)] * (Py/Px2)^(1-α)  (resultado analítico)
xC = alpha * (UA / (alpha**alpha * (1-alpha)**(1-alpha))) * (Py / Px2) ** (1 - alpha)
yC = (1 - alpha) * (UA / (alpha**alpha * (1-alpha)**(1-alpha))) * (Px2 / Py) ** alpha
MC = xC * Px2 + yC * Py   # ingreso compensado

# ── Gráfico ────────────────────────────────────────────────────────────────────
XMAX, YMAX = 18, 14
x_arr = np.linspace(0.05, XMAX, 500)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_xlabel("Cantidad de bien X", fontsize=12)
ax.set_ylabel("Cantidad de bien Y", fontsize=12)
ax.set_title("Efecto Sustitución y Efecto Ingreso", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

COLOR_A  = "#3C3489"   # púrpura  – inicial
COLOR_B  = "#0F6E56"   # teal     – nuevo
COLOR_C  = "#BA7517"   # ámbar    – compensado
COLOR_CI_A = "#7F77DD" # curva indiferencia original (más claro)
COLOR_CI_B = "#1D9E75" # curva indiferencia nueva

legend_handles = []

if show_A:
    # Curva de indiferencia original
    yi_A = curva_indiferencia(UA, x_arr)
    ax.plot(x_arr, yi_A, color=COLOR_CI_A, lw=2, alpha=0.7, label="Curva indiferencia U_A")
    # Línea presupuestaria inicial
    ax.plot([0, M/Px], [M/Py, 0], color=COLOR_A, lw=2, label="Presupuesto inicial")
    # Punto A
    ax.plot(xA, yA, "o", color=COLOR_A, markersize=9, zorder=5)
    ax.annotate("A", (xA, yA), textcoords="offset points", xytext=(8, 6),
                fontsize=12, fontweight="bold", color=COLOR_A)
    ax.plot([xA, xA], [0, yA], "--", color=COLOR_A, lw=1, alpha=0.5)
    ax.plot([0, xA], [yA, yA], "--", color=COLOR_A, lw=1, alpha=0.5)
    legend_handles += [
        Line2D([0],[0], color=COLOR_CI_A, lw=2, label="Curva de indiferencia $U_A$"),
        Line2D([0],[0], color=COLOR_A,    lw=2, label="Presupuesto inicial"),
        Line2D([0],[0], marker='o', color=COLOR_A, lw=0, markersize=8, label="Punto A (óptimo inicial)"),
    ]

if show_B:
    # Curva de indiferencia nueva
    yi_B = curva_indiferencia(UB, x_arr)
    ax.plot(x_arr, yi_B, color=COLOR_CI_B, lw=2, alpha=0.7)
    # Línea presupuestaria nueva
    ax.plot([0, M/Px2], [M/Py, 0], color=COLOR_B, lw=2)
    # Punto B
    ax.plot(xB, yB, "o", color=COLOR_B, markersize=9, zorder=5)
    ax.annotate("B", (xB, yB), textcoords="offset points", xytext=(8, 6),
                fontsize=12, fontweight="bold", color=COLOR_B)
    ax.plot([xB, xB], [0, yB], "--", color=COLOR_B, lw=1, alpha=0.5)
    ax.plot([0, xB], [yB, yB], "--", color=COLOR_B, lw=1, alpha=0.5)
    legend_handles += [
        Line2D([0],[0], color=COLOR_CI_B, lw=2, label="Curva de indiferencia $U_B$"),
        Line2D([0],[0], color=COLOR_B,    lw=2, label="Presupuesto nuevo"),
        Line2D([0],[0], marker='o', color=COLOR_B, lw=0, markersize=8, label="Punto B (nuevo óptimo)"),
    ]

if show_C and show_B:
    # Línea presupuestaria compensada (Slutsky)
    ax.plot([0, MC/Px2], [MC/Py, 0], color=COLOR_C, lw=2, ls="--")
    # Punto C
    ax.plot(xC, yC, "o", color=COLOR_C, markersize=9, zorder=5)
    ax.annotate("C", (xC, yC), textcoords="offset points", xytext=(8, 6),
                fontsize=12, fontweight="bold", color=COLOR_C)
    ax.plot([xC, xC], [0, yC], "--", color=COLOR_C, lw=1, alpha=0.5)
    # Flecha efecto sustitución: A → C
    y_arrow = min(yA, yC) - 0.8
    ax.annotate("", xy=(xC, y_arrow), xytext=(xA, y_arrow),
                arrowprops=dict(arrowstyle="->", color=COLOR_C, lw=2))
    ax.text((xA + xC) / 2, y_arrow - 0.5, "Ef. sustitución",
            ha="center", fontsize=10, color=COLOR_C, fontweight="bold")
    legend_handles += [
        Line2D([0],[0], color=COLOR_C, lw=2, ls="--", label="Presupuesto compensado"),
        Line2D([0],[0], marker='o', color=COLOR_C, lw=0, markersize=8, label="Punto C (compensado)"),
    ]

if show_EI and show_B and show_C:
    # Flecha efecto ingreso: C → B
    y_arrow2 = min(yA, yC) - 1.6
    ax.annotate("", xy=(xB, y_arrow2), xytext=(xC, y_arrow2),
                arrowprops=dict(arrowstyle="->", color=COLOR_B, lw=2))
    ax.text((xC + xB) / 2, y_arrow2 - 0.5, "Ef. ingreso",
            ha="center", fontsize=10, color=COLOR_B, fontweight="bold")

ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
          framealpha=0.9, edgecolor="#ccc")
ax.grid(True, alpha=0.2)
plt.tight_layout()

with col_graf:
    st.pyplot(fig)

# ── Resumen numérico ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Resumen de los puntos")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Punto A — x", f"{xA:.2f}")
    st.metric("Punto A — y", f"{yA:.2f}")
    st.metric("Utilidad U_A", f"{UA:.3f}")
with c2:
    st.metric("Punto B — x", f"{xB:.2f}")
    st.metric("Punto B — y", f"{yB:.2f}")
    st.metric("Utilidad U_B", f"{UB:.3f}")
with c3:
    st.metric("Punto C — x", f"{xC:.2f}")
    st.metric("Punto C — y", f"{yC:.2f}")
    st.metric("Ingreso compensado M'", f"{MC:.2f}")

st.divider()
st.subheader("Descomposición del efecto total")
delta_total = xB - xA
delta_sust  = xC - xA
delta_ing   = xB - xC

c4, c5, c6 = st.columns(3)
c4.metric("Efecto total  (A → B)",        f"Δx = {delta_total:+.2f}")
c5.metric("Efecto sustitución (A → C)",   f"Δx = {delta_sust:+.2f}")
c6.metric("Efecto ingreso  (C → B)",      f"Δx = {delta_ing:+.2f}")

st.caption(
    "Comprobación: Efecto total = Efecto sustitución + Efecto ingreso  →  "
    f"{delta_total:.2f} = {delta_sust:.2f} + {delta_ing:.2f} = {delta_sust+delta_ing:.2f} ✓"
)

# ── Instrucciones de instalación ───────────────────────────────────────────────
with st.expander("¿Cómo ejecutar esta app?"):
    st.code("""
# 1. Instalar dependencias (solo la primera vez)
pip install streamlit matplotlib numpy

# 2. Correr la app
streamlit run efecto_ingreso_sustitucion.py
    """, language="bash")
    st.markdown(
        "Se abre automáticamente en tu navegador en `http://localhost:8501`. "
        "Comparte el archivo `.py` con tus compañeros y ellos siguen los mismos pasos."
    )