import os
import shutil

from jinja2 import Environment, FileSystemLoader
import pandas as pd
import math
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# === Jinja2 ===
env = Environment(loader=FileSystemLoader("templates"))

# Paleta “modern data”
COLOR_PETROLEO = "#264653"
COLOR_MENTA    = "#2A9D8F"  # MRR
COLOR_MOSTAZA  = "#E9C46A"  # No-MRR
COLOR_CORAL    = "#F4A261"
COLOR_ROJO     = "#E76F51"
COLOR_GRIS_CL  = "#EFF2F4"

# Para títulos/estética común
COLOR_TITULO   = COLOR_PETROLEO
COLOR_TEXTO    = "#2B2F33"

def fmt_int_eu(v):
    try:
        return f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return v

def fmt_num_eu(v, dec=2):
    try:
        s = f"{v:,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return v

def fmt_pct_eu(v, dec=1):
    try:
        s = fmt_num_eu(v, dec)
        return f"{s} %"
    except Exception:
        return v

def fmt_eur_eu(v, dec=0, symbol=False):
    s = fmt_num_eu(v, dec)
    return f"{s} €" if symbol else s

# Registra filtros para usar en index.html.j2 ({{ valor|int_eu }} etc.)
env.filters["int_eu"] = fmt_int_eu          # 12.345
env.filters["num_eu"] = fmt_num_eu          # 12.345,67
env.filters["pct_eu"] = fmt_pct_eu          # 12,3 %
env.filters["eur_eu"] = fmt_eur_eu          # 12.345 € (si symbol=True)

def _safe(v):  # tolera None/NaN
    try:
        if v is None: return None
        if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    except Exception:
        return None

def fmt_int_eu_py(v):
    v = _safe(v)
    if v is None: return ""
    return f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_num_eu_py(v, dec=2):
    v = _safe(v)
    if v is None: return ""
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct_eu_py(v, dec=1):
    s = fmt_num_eu_py(v, dec)
    return f"{s} %" if s else ""

def fmt_eur_eu_py(v, dec=0, symbol=True):
    s = fmt_num_eu_py(v, dec)
    return f"{s} €" if (s and symbol) else s

# Aplicar a Series
def series_eur_eu(s: pd.Series, dec=0, symbol=True):
    return s.apply(lambda v: fmt_eur_eu_py(v, dec=dec, symbol=symbol))

def series_num_eu(s: pd.Series, dec=2):
    return s.apply(lambda v: fmt_num_eu_py(v, dec=dec))

def series_pct_eu(s: pd.Series, dec=1):
    return s.apply(lambda v: fmt_pct_eu_py(v, dec=dec))

# === Estilo común para gráficos Plotly ===
def estilizar_grafico(fig, titulo_html, y_title=None, x_title=None, showlegend=False, margins=(60,20,20,20)):
    mt, mb, ml, mr = margins
    fig.update_layout(
        template='simple_white',
        title=dict(text=titulo_html, x=0.5, xanchor='center', font=dict(size=16, color=COLOR_TITULO)),
        font=dict(size=12, color=COLOR_TEXTO),
        yaxis_title=y_title,
        xaxis_title=x_title,
        showlegend=showlegend,
        margin=dict(t=mt, b=mb, l=ml, r=mr),
        hoverlabel=dict(bgcolor='white', font_color=COLOR_TEXTO, bordercolor=COLOR_GRIS_CL)
    )

# === Hover para los graficos ===
def set_hover_eu(fig, df: pd.DataFrame, campos, plantilla):
    cd = df[campos].to_numpy()  # (n, k)
    fig.update_traces(customdata=cd, hovertemplate=plantilla)
    return fig

# === Formateo europeo para texto dentro de barras (Python) ===
def eur_eu_py(v, dec=0, symbol=True):
    try:
        s = f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} €" if symbol else s
    except Exception:
        return v

print("🟡 Inicio de la generación de la web")

print("🟡 Preparando la carpeta de salida...")
# Elimina la carpeta 'output' si ya existe para evitar errores y duplicados
if os.path.exists("output"):
    shutil.rmtree("output")
# Crea una nueva carpeta 'output'
os.makedirs("output")
# Copia toda la carpeta 'static' dentro de la nueva carpeta 'output'
shutil.copytree("static", "output/static")
print("🟢 Carpeta 'static' copiada a 'output/static'")


# === 1. Cargar datos ===
csv_path = Path("data/empresas_grandes_beneficiarios_limpio2025.csv")
df_empresas = pd.read_csv(csv_path)
print("🟢 Datos cargados:", df_empresas.shape)

# === 2. Calcular indicadores principales ===
n_beneficiarios = df_empresas["NOMBRE"].nunique()
total_equivalente = df_empresas["AYUDA EQUIVALENTE ACUMULADA TOTAL"].sum()
total_mrr = df_empresas["AYUDA EQUIVALENTE DEL MRR"].sum()
pct_mrr = total_mrr / total_equivalente if total_equivalente else 0

# === 2.1 Gráfico circular: MRR vs No-MRR ===
total_no_mrr = total_equivalente - total_mrr

df_mrr_vs_no = pd.DataFrame({
    "Concepto": ["MRR", "No MRR"],
    "Total": [total_mrr, total_no_mrr]
})
df_mrr_vs_no["Total_eu"] = series_eur_eu(df_mrr_vs_no["Total"], dec=0, symbol=True)

colores_mrr_map = {
    "MRR": "#2A9D8F",    # menta
    "No MRR": "#E9C46A"  # mostaza
}
fig_mrr_vs_no = px.pie(
    df_mrr_vs_no,
    names="Concepto",
    values="Total",
    color="Concepto",
    color_discrete_map=colores_mrr_map
)
fig_mrr_vs_no.update_traces(
    textinfo='percent',
    textposition='inside',
    marker=dict(line=dict(color="white", width=1)),
)
# Hover:
set_hover_eu(
    fig_mrr_vs_no,
    df_mrr_vs_no,
    campos=["Total_eu"],
    plantilla="<b>%{label}</b> %{customdata[0]}<extra></extra>"
)

estilizar_grafico(
    fig_mrr_vs_no,
    "<b>Distribución de ayudas: MRR vs No-MRR</b>",
    showlegend=False,
    margins=(60, 20, 20, 20)
)

grafico_mrr_vs_no_file = "grafico_pie_mrr_vs_no.html"
fig_mrr_vs_no.write_html(
    f"output/{grafico_mrr_vs_no_file}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 3. Resumen por tipo de entidad ===
resumen_por_tipo = (
    df_empresas
    .groupby("TIPO_ENTIDAD")
    .agg(
        total_ayuda   = ("AYUDA EQUIVALENTE ACUMULADA TOTAL", "sum"),
        num_empresas  = ("NIF", "count"),
        media_empresa = ("AYUDA EQUIVALENTE ACUMULADA TOTAL", "mean")
    )
    .sort_values(by="num_empresas", ascending=False)
)

# Calcular % del total
suma_total = resumen_por_tipo["total_ayuda"].sum()
resumen_por_tipo["%_del_total"] = resumen_por_tipo["total_ayuda"] / suma_total * 100

# Convertir a lista de dicts para Jinja (primera tabla)
resumen_por_tipo_list = resumen_por_tipo.reset_index().to_dict(orient="records")

# === 4.0 Asignar paleta y mapeo de colores por tipo entidad (colores fijos) ===
colores_tipos_map = {
    "Sociedad de Responsabilidad Limitada (S.L.)": COLOR_PETROLEO,
    "Sociedad Anónima (S.A.)": COLOR_MENTA,
    "Sociedades Cooperativas": COLOR_CORAL,
    "Comunidad de Bienes": COLOR_MOSTAZA,
    "Sociedad Comanditaria": "#B8C4CC"
}

# === 4.1 Generar gráfico de tarta ===
df_tarta = resumen_por_tipo.reset_index()
df_tarta["total_eu"] = series_eur_eu(df_tarta["total_ayuda"], dec=0, symbol=True)

fig_tarta = px.pie(
    df_tarta,
    names="TIPO_ENTIDAD",
    values="%_del_total",
    color="TIPO_ENTIDAD",
    color_discrete_map=colores_tipos_map
)
fig_tarta.update_traces(
    textinfo='percent',
    textposition='inside',
    marker=dict(line=dict(color="white", width=1)),
)
estilizar_grafico(
    fig_tarta,
    "<b>Distribución porcentual de ayudas por tipo de entidad</b>",
    showlegend=True,
    margins=(60, 20, 20, 20)
)
fig_tarta.update_layout(
    legend=dict(title="Tipo de Entidad", font=dict(size=10), bgcolor="rgba(255,255,255,0)", borderwidth=0)
)
set_hover_eu(
    fig_tarta,
    df_tarta,
    campos=["total_eu"],
    plantilla="<b>%{label}</b><br>Total ayuda: %{customdata[0]}<extra></extra>"
)
grafico_tarta_file = "grafico_pie_tipo_entidad.html"
fig_tarta.write_html(
    f"output/{grafico_tarta_file}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 5. Histograma de ayuda media por tipo (ordenado desc) ===
df_hist = resumen_por_tipo.reset_index().copy()

# Orden para el eje X según la media (descendente)
orden_tipos_por_media = (
    df_hist.sort_values("media_empresa", ascending=False)["TIPO_ENTIDAD"].tolist()
)

# Texto formateado para mostrar encima de la barra
df_hist["media_empresa_eu"] = series_eur_eu(df_hist["media_empresa"], dec=0, symbol=True)

fig_hist = px.bar(
    df_hist,
    x="TIPO_ENTIDAD",
    y="media_empresa",
    color="TIPO_ENTIDAD",
    color_discrete_map=colores_tipos_map,
    text=df_hist["media_empresa_eu"],  # etiqueta sobre barra
    category_orders={"TIPO_ENTIDAD": orden_tipos_por_media}
)
fig_hist.update_traces(textposition='outside', cliponaxis=False)

# Hover:
set_hover_eu(
    fig_hist,
    df_hist,
    campos=["TIPO_ENTIDAD", "media_empresa_eu"],
    plantilla="<b>%{customdata[0]}</b><br>Media por empresa: %{customdata[1]}<extra></extra>"
)

estilizar_grafico(
    fig_hist,
    "<b>Ayuda media por tipo de entidad</b>",
    y_title="Ayuda media (€)",
    x_title="Tipo de entidad",
    showlegend=False,
    margins=(60, 20, 20, 20)
)
fig_hist.update_layout(
    yaxis=dict(automargin=True, range=[0, df_hist["media_empresa"].max() * 1.15])
)

nombre_archivo_hist = "histograma_ayuda_media.html"
fig_hist.write_html(
    f"output/{nombre_archivo_hist}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 6. Top 10 empresas por ayuda acumulada (1 fila = 1 empresa) ===
top10_empresas_df = (
    df_empresas
    .loc[:, ["NOMBRE", "AYUDA EQUIVALENTE ACUMULADA TOTAL"]]
    .sort_values("AYUDA EQUIVALENTE ACUMULADA TOTAL", ascending=False)
    .head(10)
    .rename(columns={"NOMBRE": "Empresa", "AYUDA EQUIVALENTE ACUMULADA TOTAL": "Total Ayuda (€)"})
)
top10_empresas_list = top10_empresas_df.to_dict(orient="records")  # para Jinja

# Histograma Top 10
orden_empresas = top10_empresas_df["Empresa"].tolist()

# Etiqueta sobre barra con notación europea + €
top10_empresas_df["Total_eu"] = series_eur_eu(top10_empresas_df["Total Ayuda (€)"], dec=0, symbol=True)

fig_top10_hist = px.bar(
    top10_empresas_df,
    x="Empresa",
    y="Total Ayuda (€)",
    text=top10_empresas_df["Total_eu"],   # etiqueta visible encima
    category_orders={"Empresa": orden_empresas},
    color_discrete_sequence=[COLOR_PETROLEO]
)
fig_top10_hist.update_traces(textposition='outside', cliponaxis=False)

# Hover:
set_hover_eu(
    fig_top10_hist,
    top10_empresas_df,
    campos=["Empresa", "Total_eu"],
    plantilla="<b>%{customdata[0]}</b><br>Total acumulado: %{customdata[1]}<extra></extra>"
)

estilizar_grafico(
    fig_top10_hist,
    "<b>Top 10 empresas por ayuda acumulada</b>",
    y_title="Ayuda acumulada (€)",
    x_title="Empresa",
    showlegend=False,
    margins=(60, 100, 20, 20)
)
fig_top10_hist.update_layout(
    xaxis_tickangle=-45,
    yaxis=dict(range=[0, top10_empresas_df["Total Ayuda (€)"].max() * 1.15])
)

nombre_archivo_top10_hist = "histograma_top10_empresas.html"
fig_top10_hist.write_html(
    f"output/{nombre_archivo_top10_hist}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 6.2 Métricas para el texto dinámico bajo el Top 10 ===
suma_top10 = top10_empresas_df["Total Ayuda (€)"].sum()
pct_top10 = (suma_top10 / total_equivalente * 100) if total_equivalente else 0

# === 7. Curva de Lorenz e índice de Gini (con métricas) ===
valores = df_empresas["AYUDA EQUIVALENTE ACUMULADA TOTAL"].to_numpy(dtype=float)
valores_ordenados = np.sort(valores)
n = len(valores_ordenados)
suma_total_emp = valores_ordenados.sum()

# Proporciones acumuladas
x_lorenz_base = np.linspace(0, 1, n, endpoint=True)
y_acum = np.cumsum(valores_ordenados) / suma_total_emp
x_lorenz = np.insert(x_lorenz_base, 0, 0.0)
y_lorenz = np.insert(y_acum, 0, 0.0)

# Índice de Gini
area = np.trapz(y_lorenz, x_lorenz)
gini = 1 - 2 * area

# Puntos 50% y 80%
idx50 = np.searchsorted(y_lorenz, 0.5)
idx80 = np.searchsorted(y_lorenz, 0.8)
x50 = float(x_lorenz[idx50]) if idx50 < len(x_lorenz) else None
x80 = float(x_lorenz[idx80]) if idx80 < len(x_lorenz) else None

# KPIs de concentración: 50% del total
pct_empresas_50 = (x50 * 100) if x50 is not None else None
n_empresas_total = len(df_empresas)
n_empresas_50 = int(round(x50 * n_empresas_total)) if x50 is not None else None

# Minoría que concentra la mitad
pct_empresas_50_resto = (100 - pct_empresas_50) if pct_empresas_50 is not None else None  # ~3,3
n_empresas_50_minoria = (n_empresas_total - n_empresas_50) if n_empresas_50 is not None else None

# Figura Plotly Lorenz
fig_lorenz = go.Figure()
fig_lorenz.add_trace(go.Scatter(
    x=x_lorenz, y=y_lorenz,
    mode="lines",
    name="Curva de Lorenz",
    line=dict(width=2, color="#336699")
))
fig_lorenz.add_trace(go.Scatter(
    x=[0,1], y=[0,1],
    mode="lines",
    name="Igualdad perfecta",
    line=dict(width=1, dash="dash", color="#999999"),
    hoverinfo="skip"
))
# Guías 50% y 80%
if x50 is not None:
    fig_lorenz.add_trace(go.Scatter(x=[x50, x50], y=[0, 0.5], mode="lines",
                                    line=dict(color="#cc0000", dash="dot"), showlegend=False, hoverinfo="skip"))
    fig_lorenz.add_trace(go.Scatter(x=[0, x50], y=[0.5, 0.5], mode="lines",
                                    line=dict(color="#cc0000", dash="dot"), showlegend=False, hoverinfo="skip"))
if x80 is not None:
    fig_lorenz.add_trace(go.Scatter(x=[x80, x80], y=[0, 0.8], mode="lines",
                                    line=dict(color="#2e7d32", dash="dot"), showlegend=False, hoverinfo="skip"))
    fig_lorenz.add_trace(go.Scatter(x=[0, x80], y=[0.8, 0.8], mode="lines",
                                    line=dict(color="#2e7d32", dash="dot"), showlegend=False, hoverinfo="skip"))

estilizar_grafico(
    fig_lorenz,
    f"<b>Curva de Lorenz – Subvenciones 2024</b><br><span style='font-size:0.9em'>Índice de Gini = {gini:.4f}</span>",
    y_title="Proporción acumulada de subvenciones",
    x_title="Proporción acumulada de empresas",
    showlegend=True,
    margins=(80, 60, 40, 20)
)
fig_lorenz.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2), yaxis=dict(range=[0,1]))

nombre_archivo_lorenz = "curva_lorenz_gini.html"
fig_lorenz.write_html(
    f"output/{nombre_archivo_lorenz}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 8. Top 10 empresas por AYUDA EQUIVALENTE DEL MRR  y OTRAS ======================

# 8.1 Columna “Ayuda no MRR (€)” calculada
df_empresas["Ayuda no MRR (€)"] = (
    df_empresas["AYUDA EQUIVALENTE ACUMULADA TOTAL"]
    - df_empresas["AYUDA EQUIVALENTE DEL MRR"]
).clip(lower=0)

top10_no_mrr_empresas_df = (
    df_empresas
    .loc[:, ["NOMBRE", "Ayuda no MRR (€)"]]
    .rename(columns={"NOMBRE": "Empresa"})
    .query("`Ayuda no MRR (€)` > 0")
    .sort_values("Ayuda no MRR (€)", ascending=False)
    .head(10)
    .copy()
)

# 8.2 Top 10 MRR (tabla)
top10_mrr_empresas_df = (
    df_empresas
    .loc[:, ["NOMBRE", "AYUDA EQUIVALENTE DEL MRR"]]
    .rename(columns={
        "NOMBRE": "Empresa",
        "AYUDA EQUIVALENTE DEL MRR": "Ayuda MRR (€)"
    })
    .query("`Ayuda MRR (€)` > 0")
    .sort_values("Ayuda MRR (€)", ascending=False)
    .head(10)
    .copy()
)

# Debug para comprobar que hay datos y las claves son correctas
print("[DEBUG] top10_no_mrr_empresas_df cols:", top10_no_mrr_empresas_df.columns.tolist())
print("[DEBUG] top10_no_mrr_empresas_df head:\n", top10_no_mrr_empresas_df.head(3))

top10_mrr_empresas_list = top10_mrr_empresas_df.to_dict(orient="records")

# 8.3 Gráfico Top 10 MRR
top10_mrr_empresas_df["Ayuda_MRR_fmt"] = top10_mrr_empresas_df["Ayuda MRR (€)"].apply(lambda v: eur_eu_py(v, 0, True))
orden_empresas_mrr = top10_mrr_empresas_df["Empresa"].tolist()

fig_top10_mrr = px.bar(
    top10_mrr_empresas_df,
    x="Empresa",
    y="Ayuda MRR (€)",
    text=top10_mrr_empresas_df["Ayuda_MRR_fmt"],
    category_orders={"Empresa": orden_empresas_mrr},
    color_discrete_sequence=["#2A9D8F"]
)

fig_top10_mrr.update_traces(
    textposition="outside",
    cliponaxis=False,
    customdata=top10_mrr_empresas_df[["Empresa", "Ayuda_MRR_fmt"]].to_numpy(),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
)
estilizar_grafico(
    fig_top10_mrr,
    "<b>Top 10 empresas por ayuda MRR</b>",
    y_title="Ayuda MRR (€)",
    x_title="Empresa",
    showlegend=False,
    margins=(60, 100, 20, 20)
)
fig_top10_mrr.update_layout(
    xaxis_tickangle=-45,
    yaxis=dict(range=[0, top10_mrr_empresas_df["Ayuda MRR (€)"].max() * 1.15])
)
nombre_archivo_top10_mrr = "histograma_top10_mrr.html"
fig_top10_mrr.write_html(
    f"output/{nombre_archivo_top10_mrr}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# 8.4 Top 10 no MRR (tabla)
top10_no_mrr_empresas_df = (
    df_empresas
    .loc[:, ["NOMBRE", "Ayuda no MRR (€)"]]
    .rename(columns={"NOMBRE": "Empresa"})
    .query("`Ayuda no MRR (€)` > 0")
    .sort_values("Ayuda no MRR (€)", ascending=False)
    .head(10)
    .copy()
)
top10_no_mrr_empresas = top10_no_mrr_empresas_df.to_dict(orient="records")

# 8.5 Gráfico Top 10 no MRR
top10_no_mrr_empresas_df["Ayuda_no_MRR_fmt"] = top10_no_mrr_empresas_df["Ayuda no MRR (€)"].apply(lambda v: eur_eu_py(v, 0, True))
orden_empresas_no_mrr = top10_no_mrr_empresas_df["Empresa"].tolist()

fig_top10_no_mrr = px.bar(
    top10_no_mrr_empresas_df,
    x="Empresa",
    y="Ayuda no MRR (€)",
    text=top10_no_mrr_empresas_df["Ayuda_no_MRR_fmt"],
    category_orders={"Empresa": orden_empresas_no_mrr},
    color_discrete_sequence=["#E9C46A"]
)
fig_top10_no_mrr.update_traces(
    textposition="outside",
    cliponaxis=False,
    customdata=top10_no_mrr_empresas_df[["Empresa", "Ayuda_no_MRR_fmt"]].to_numpy(),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
)
estilizar_grafico(
    fig_top10_no_mrr,
    "<b>Top 10 por ayudas no MRR</b>",
    y_title="Ayuda no MRR (€)",
    x_title="Empresa",
    showlegend=False,
    margins=(60, 100, 20, 20)
)
fig_top10_no_mrr.update_layout(
    xaxis_tickangle=-45,
    yaxis=dict(range=[0, top10_no_mrr_empresas_df["Ayuda no MRR (€)"].max() * 1.15])
)
histograma_top10_no_mrr = "histograma_top10_no_mrr.html"
fig_top10_no_mrr.write_html(
    f"output/{histograma_top10_no_mrr}",
    include_plotlyjs="cdn",
    config={"locale": "es"}
)

# === 9. Renderizar plantilla ===
template = env.get_template("index.html.j2")

html_output = template.render(
    # KPIs
    n_beneficiarios            = n_beneficiarios,
    total_equivalente          = total_equivalente,
    total_mrr                  = total_mrr,
    pct_mrr                    = pct_mrr,
    # Grafico 
    grafico_mrr_vs_no = grafico_mrr_vs_no_file,
    # Tabla 1 (resumen por tipo)
    resumen_tipo_entidad       = resumen_por_tipo_list,
    # Gráficos
    grafico_tarta_tipo_entidad = grafico_tarta_file,
    histograma_ayuda_media     = nombre_archivo_hist,
    # Tabla 2 (Top 10) como datos
    top10_empresas             = top10_empresas_list,
    # Gráfico Top 10
    histograma_top10           = nombre_archivo_top10_hist,
    # Porcentaje ayudas Top 10 y 50%
    pct_top10                  = pct_top10,
    pct_empresas_50            = pct_empresas_50,
    n_empresas_50              = n_empresas_50,
    pct_empresas_50_resto      = pct_empresas_50_resto,
    n_empresas_50_minoria      = n_empresas_50_minoria,
    # Curva Lorenz e índice Gini
    curva_lorenz_gini          = nombre_archivo_lorenz,
    # Top 10 MRR
    top10_mrr_empresas         = top10_mrr_empresas_list,
    histograma_top10_mrr       = nombre_archivo_top10_mrr,
    # Histograma top 10 otras ayudas
    top10_no_mrr_empresas      = top10_no_mrr_empresas,
    histograma_top10_no_mrr    = histograma_top10_no_mrr,
    )


# === 10. Guardar HTML ===
Path("output/index.html").write_text(html_output, encoding="utf-8")
print("✅ Web generada correctamente en output/index.html")