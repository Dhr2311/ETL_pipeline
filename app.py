import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import time


load_dotenv()
ENGINE_URL = os.environ["DATABASE_URL"]
engine = create_engine(ENGINE_URL, future=True)

# ------------- Data loaders -------------
def load_data():
    df = pd.read_sql("SELECT * FROM encounters_master", engine)
    dq = pd.read_sql("SELECT * FROM dq_issues", engine)
    # types
    if "recorded_at_utc" in df:
        df["recorded_at_utc"] = pd.to_datetime(df["recorded_at_utc"], errors="coerce", utc=True)
    return df, dq

df, dq = load_data()

# ------------- App -------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Hospital Dashboard"
server = app.server

def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([html.H6(title), html.H3(value, className="card-title")]),
        className="mb-3"
    )

def overview_layout(df, dq):
    # KPIs
    rows = len(df)
    distinct_enc = df["encounterid"].nunique() if "encounterid" in df else 0
    invalid_codes = int((df.get("code_valid") == False).sum()) if "code_valid" in df else 0
    future_ts = int((df.get("is_future") == True).sum()) if "is_future" in df else 0

    # Charts
    figs = []
    if "recorded_at_utc" in df and df["recorded_at_utc"].notna().any():
        ts = df.assign(day=df["recorded_at_utc"].dt.date).groupby("day").size().reset_index(name="count")
        figs.append(px.bar(ts, x="day", y="count", title="Encounters per day"))
    if "code" in df:
        top_codes = df["code"].value_counts().head(20).reset_index()
        top_codes.columns = ["code","count"]
        figs.append(px.bar(top_codes, x="code", y="count", title="Top ICD codes"))
    if "encountertype" in df:
        enc_counts = df["encountertype"].value_counts().reset_index()
        enc_counts.columns = ["encountertype","count"]
        figs.append(px.pie(enc_counts, names="encountertype", values="count", title="Encounter types"))
    if "weight_kg" in df and "height_cm" in df:
        figs.append(px.scatter(df, x="height_cm", y=df["weight_kg"].fillna(df.get("weight_pred")),
                               color=df["weight_kg"].isna().map({True:"Imputed", False:"Observed"}),
                               title="Weight vs Height (Imputed vs Observed)"))

    kpis = dbc.Row([
        dbc.Col(kpi_card("Rows", rows), md=3),
        dbc.Col(kpi_card("Distinct Encounters", distinct_enc), md=3),
        dbc.Col(kpi_card("Invalid Codes", invalid_codes), md=3),
        dbc.Col(kpi_card("Future Timestamps", future_ts), md=3),
    ])

    charts = [dcc.Graph(figure=f) for f in figs]
    return dbc.Container([kpis, html.Div(charts)], fluid=True)

def dq_layout(dq):
    reason_counts = dq["reason"].value_counts().reset_index()
    reason_counts.columns = ["reason","count"]
    fig = px.bar(reason_counts, x="reason", y="count", title="DQ Issues by Type")

    return dbc.Container([
        dcc.Graph(figure=fig),
        html.Hr(),
        dash_table.DataTable(
            id="dq-table",
            data=dq.to_dict("records"),
            columns=[{"name": c, "id": c} for c in dq.columns],
            page_size=15,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto"},
        )
    ], fluid=True)

# --- Records tab (search + table + details) ---
def records_layout(df):
    # Search input
    search_bar = dbc.InputGroup([
        dbc.Input(id="search-name", placeholder="Search given/family name...", type="text"),
        dbc.Button("Search", id="btn-search", n_clicks=0)
    ], className="mb-3")

    # Patient table (render patientid as link-like cell)
    columns = [{"name": c, "id": c} for c in df.columns]
    table = dash_table.DataTable(
        id="patients-table",
        data=df.to_dict("records"),
        columns=columns,
        page_size=12,
        filter_action="native",
        sort_action="native",
        row_selectable=False,
        cell_selectable=True,
        style_table={"overflowX":"auto", "minHeight":"400px"},
        style_cell={"fontFamily":"Inter, system-ui", "fontSize":"14px"},
    )

    # Details panel
    details = dbc.Card([
        dbc.CardHeader("Patient Record"),
        dbc.CardBody(id="record-card", children="Double-click a patientid to open full record."),
    ])

    return dbc.Container([
        search_bar,
        dbc.Row([
            dbc.Col(table, md=8),
            dbc.Col(details, md=4),
        ]),
        # stores for double-click detection
        dcc.Store(id="last-click"),
        dcc.Store(id="dblclick-patientid")
    ], fluid=True)

app.layout = dbc.Container([
    html.H2("Hospital Management Dashboard"),
    dcc.Tabs(id="tabs", value="tab-overview", children=[
        dcc.Tab(label="Overview", value="tab-overview"),
        dcc.Tab(label="Data Quality", value="tab-dq"),
        dcc.Tab(label="Records", value="tab-records"),
    ]),
    html.Div(id="tab-content")
], fluid=True)

# ------------- Callbacks -------------
@app.callback(Output("tab-content","children"),
              Input("tabs","value"))
def render_tab(tab):
    global df, dq
    if tab == "tab-overview":
        return overview_layout(df, dq)
    if tab == "tab-dq":
        return dq_layout(dq)
    return records_layout(df)

# Search by name (givenname/familyname contains)
@app.callback(
    Output("patients-table","data"),
    Input("btn-search","n_clicks"),
    State("search-name","value")
)
def run_search(n, q):
    if not n:
        return df.to_dict("records")
    if not q:
        return df.to_dict("records")
    qlow = q.strip().lower()
    mask = (
        df.get("givenname","").astype(str).str.lower().str.contains(qlow, na=False) |
        df.get("familyname","").astype(str).str.lower().str.contains(qlow, na=False)
    )
    return df.loc[mask].to_dict("records")

# "Double-click" detection on patientid:
# We use DataTable active_cell; if the same patientid is clicked twice within 500ms, treat as double-click.
@app.callback(
    Output("dblclick-patientid","data"),
    Output("last-click","data"),
    Input("patients-table","active_cell"),
    State("patients-table","data"),
    State("last-click","data"),
    prevent_initial_call=True
)
def detect_double_click(active_cell, rows, last_click):
    if not active_cell or not rows: return dash.no_update, last_click
    r, c = active_cell["row"], active_cell["column_id"]
    if c != "patientid":  # only when patientid cell is clicked
        return dash.no_update, last_click

    pid = rows[r].get("patientid")
    t = time.time()
    if last_click and last_click.get("pid") == pid and (t - last_click.get("ts", 0)) < 0.5:
        # double click!
        return {"pid": pid}, {"pid": pid, "ts": t}
    # first click
    return dash.no_update, {"pid": pid, "ts": t}

# Render full record on double-click
@app.callback(
    Output("record-card","children"),
    Input("dblclick-patientid","data")
)
def show_record(data):
    if not data: 
        return "Double-click a patientid to open full record."
    pid = data["pid"]
    recs = df.loc[df["patientid"] == pid]
    if recs.empty:
        return f"No record found for {pid}"
    # Show the most recent encounter
    rec = recs.sort_values("recorded_at_utc").iloc[-1] if "recorded_at_utc" in recs else recs.iloc[-1]
    items = []
    for k, v in rec.items():
        items.append(html.Div([html.Strong(f"{k}: "), html.Span(str(v))]))
    return items

if __name__ == "__main__":
    app.run(debug=True, port=8050)
