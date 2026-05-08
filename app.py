"""
Interactive dashboard for the GraphRAG Link Prediction project.
42913 Social and Information Network Analysis — Group 4

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── project path ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── file paths ──────────────────────────────────────────────────────────────
EDGES_PATH       = PROJECT_ROOT / "data" / "processed" / "graph_edges.csv"
TRAIN_PATH       = PROJECT_ROOT / "data" / "splits" / "train_edges.csv"
TEST_PATH        = PROJECT_ROOT / "data" / "splits" / "test_edges.csv"
SUMMARY_PATH     = PROJECT_ROOT / "results" / "metrics" / "evaluation_summary.csv"
PER_SOURCE_PATH  = PROJECT_ROOT / "results" / "metrics" / "per_source_precision.csv"
TOPK_PATH        = PROJECT_ROOT / "results" / "predictions" / "topk_predictions.csv"
SCORED_PATH      = PROJECT_ROOT / "results" / "predictions" / "scored_candidates.csv"

METHOD_COLOURS = {
    "personalized_pagerank": "#16a34a",
    "common_neighbors":      "#2563eb",
    "adamic_adar":           "#7c3aed",
    "jaccard":               "#d97706",
    "katz":                  "#f59e0b",
}
METHOD_LABELS = {
    "personalized_pagerank": "Personalised PageRank",
    "common_neighbors":      "Common Neighbours",
    "adamic_adar":           "Adamic / Adar",
    "jaccard":               "Jaccard",
    "katz":                  "Katz Index",
}

# ── Dataset 2 (HotpotQA) hardcoded results ──────────────────────────────────
HOTPOTQA_RESULTS = pd.DataFrame([
    {"method": "personalized_pagerank", "auc_roc": 0.8084, "average_precision": 0.2228},
    {"method": "katz",                  "auc_roc": 0.7252, "average_precision": 0.0666},
    {"method": "jaccard",               "auc_roc": 0.6295, "average_precision": 0.0273},
    {"method": "adamic_adar",           "auc_roc": 0.6158, "average_precision": 0.0289},
])

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Link Prediction Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1e3a8a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: white; border-radius: 10px; padding: 18px 20px;
    border: 1px solid #e2e8f0; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #1e3a8a; }
  .metric-card .lbl { font-size: .82rem; color: #64748b; margin-top: 4px; }
  .status-ok   { color: #16a34a; font-weight: 600; }
  .status-miss { color: #dc2626; font-weight: 600; }
  h1 { color: #1e3a8a !important; }
  h2, h3 { color: #1e40af !important; }
</style>
""", unsafe_allow_html=True)

# ── cached loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_edges() -> pd.DataFrame | None:
    if not EDGES_PATH.exists():
        return None
    return pd.read_csv(EDGES_PATH).astype(int)

@st.cache_data
def load_graph() -> nx.DiGraph | None:
    edges = load_edges()
    if edges is None:
        return None
    return nx.from_pandas_edgelist(edges, source="source", target="target",
                                   create_using=nx.DiGraph())

@st.cache_data
def load_summary() -> pd.DataFrame | None:
    if not SUMMARY_PATH.exists():
        return None
    return pd.read_csv(SUMMARY_PATH)

@st.cache_data
def load_per_source() -> pd.DataFrame | None:
    if not PER_SOURCE_PATH.exists():
        return None
    return pd.read_csv(PER_SOURCE_PATH)

@st.cache_data
def load_topk() -> pd.DataFrame | None:
    if not TOPK_PATH.exists():
        return None
    return pd.read_csv(TOPK_PATH)

@st.cache_data
def load_splits() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train = pd.read_csv(TRAIN_PATH).astype(int) if TRAIN_PATH.exists() else None
    test  = pd.read_csv(TEST_PATH).astype(int)  if TEST_PATH.exists()  else None
    return train, test

# ── sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.markdown("## 🕸️ GraphRAG Dashboard")
st.sidebar.markdown("**42913 · Group 4**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🕸️ Network Explorer", "📊 Results & Algorithms",
     "🔄 Cross-Dataset", "🔍 Prediction Explorer"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset 1** — WikiVote")
st.sidebar.markdown("7,115 nodes · 103,689 edges")
st.sidebar.markdown("P@10 best: **0.3369** (3.7× random)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset 2** — HotpotQA")
st.sidebar.markdown("200 QA examples · ~23 nodes/graph")
st.sidebar.markdown("AUC best: **0.8084** (+61% random)")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("GraphRAG and Context Retrieval")
    st.markdown(
        "**42913 Social and Information Network Analysis — Topic 2 · Group 4**\n\n"
        "A flexible GraphRAG retrieval engine evaluated on **two datasets**. "
        "The same `GraphRAGPipeline` runs on both — only the dataset adapter changes."
    )

    tab1, tab2 = st.tabs(["📊 Dataset 1 — Wikipedia Vote Network", "🧠 Dataset 2 — HotpotQA"])

    with tab1:
        st.caption("Social trust graph · Link prediction · Precision@10")
    # ── Key metric cards ────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, val, lbl in [
        (col1, "7,115",   "Nodes (users)"),
        (col2, "103,689", "Directed edges"),
        (col3, "82,951",  "Training edges"),
        (col4, "20,738",  "Test edges"),
        (col5, "0.3369",  "Best P@10 (PPR)"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Pipeline status ─────────────────────────────────────────────────────
    st.subheader("Pipeline Status")
    files = {
        "Raw dataset":        PROJECT_ROOT / "data" / "raw" / "wiki-Vote.txt",
        "Processed edges":    EDGES_PATH,
        "Train split":        TRAIN_PATH,
        "Test split":         TEST_PATH,
        "Evaluation summary": SUMMARY_PATH,
        "Top-K predictions":  TOPK_PATH,
    }
    cols = st.columns(3)
    for i, (label, path) in enumerate(files.items()):
        exists = path.exists()
        icon   = "✅" if exists else "❌"
        status = "Ready" if exists else "Missing — run pipeline"
        cols[i % 3].markdown(f"{icon} **{label}**  \n<small>{status}</small>",
                             unsafe_allow_html=True)

    # ── Results table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Experiment Results — Precision@10")
    summary = load_summary()
    if summary is not None:
        display = summary.copy()
        display["method"] = display["method"].map(
            lambda m: METHOD_LABELS.get(m, m)
        )
        display = display.rename(columns={
            "method": "Method",
            "k": "K",
            "mean_precision_at_k": "Mean P@K",
            "std_precision_at_k":  "Std Dev",
            "total_hits":          "Total Hits",
            "evaluated_sources":   "Nodes Evaluated",
        })
        display["Mean P@K"] = display["Mean P@K"].map("{:.4f}".format)
        display["Std Dev"]  = display["Std Dev"].map("{:.4f}".format)
        display["vs Random"] = ["3.7×", "3.3×", "3.3×", "3.2×"][: len(display)]

        left, right = st.columns([3, 2])
        left.dataframe(display.set_index("Method"), use_container_width=True)

        fig = px.bar(
            summary,
            x="method", y="mean_precision_at_k",
            error_y="std_precision_at_k",
            color="method",
            color_discrete_map=METHOD_COLOURS,
            labels={"method": "", "mean_precision_at_k": "Mean Precision@10"},
            title="Mean Precision@10 by Method",
        )
        fig.update_layout(showlegend=False, height=320,
                          xaxis_ticktext=list(METHOD_LABELS.values()),
                          xaxis_tickvals=list(METHOD_LABELS.keys()))
        fig.add_hline(y=0.091, line_dash="dash", line_color="gray",
                      annotation_text="Random baseline (0.091)",
                      annotation_position="top right")
        right.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python scripts/run_experiments.py` to generate results.")

    with tab2:
        st.caption("Entity co-occurrence graphs · Context retrieval · AUC-ROC")
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, "200",   "QA examples"),
            (c2, "~23",   "Avg nodes / graph"),
            (c3, "0.8084","Best AUC-ROC (PPR)"),
            (c4, "+61%",  "PPR vs random AUC"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        left2, right2 = st.columns([3, 2])
        d2_disp = HOTPOTQA_RESULTS.copy()
        d2_disp["Method"] = d2_disp["method"].map(lambda m: METHOD_LABELS.get(m, m))
        d2_disp["AUC-ROC"] = d2_disp["auc_roc"].map("{:.4f}".format)
        d2_disp["Avg Precision"] = d2_disp["average_precision"].map("{:.4f}".format)
        left2.dataframe(d2_disp[["Method","AUC-ROC","Avg Precision"]].set_index("Method"),
                        use_container_width=True)
        fig2 = px.bar(
            HOTPOTQA_RESULTS, x="method", y="auc_roc",
            color="method", color_discrete_map=METHOD_COLOURS,
            labels={"method": "", "auc_roc": "AUC-ROC"},
            title="AUC-ROC by Method — HotpotQA",
        )
        fig2.update_layout(showlegend=False, height=280,
                           xaxis_ticktext=[METHOD_LABELS.get(m,m) for m in HOTPOTQA_RESULTS["method"]],
                           xaxis_tickvals=list(HOTPOTQA_RESULTS["method"]))
        fig2.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       annotation_text="Random AUC ≈ 0.500",
                       annotation_position="top right")
        right2.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — NETWORK EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ Network Explorer":
    st.title("Network Explorer")
    st.markdown(
        "Explore the structure of the Wikipedia Vote Network. "
        "The full graph has 7,115 nodes — here we focus on the most-trusted hub nodes "
        "and their immediate connections to illustrate the hub-and-spoke structure."
    )

    graph = load_graph()
    if graph is None:
        st.error("No graph data found. Run `python scripts/preprocess_data.py` first.")
        st.stop()

    # ── Controls ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    n_hubs  = col_a.slider("Number of hub nodes", 3, 15, 6)
    n_neigh = col_b.slider("Neighbours per hub", 4, 20, 8)
    seed    = col_c.slider("Random seed (layout variation)", 1, 100, 42)

    # ── Build subgraph ───────────────────────────────────────────────────────
    @st.cache_data
    def make_subgraph(n_hubs: int, n_neigh: int, seed: int):
        rng = random.Random(seed)
        in_deg = sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)
        hubs = [n for n, _ in in_deg[:n_hubs]]
        selected = set(hubs)
        for h in hubs:
            preds = list(graph.predecessors(h))
            succs = list(graph.successors(h))
            selected.update(rng.sample(preds, min(n_neigh, len(preds))))
            selected.update(rng.sample(succs, min(n_neigh // 2, len(succs))))
        sub = graph.subgraph(selected).copy()
        pos = nx.spring_layout(sub, seed=seed, k=2.0, iterations=60)
        return sub, hubs, pos

    sub, hubs, pos = make_subgraph(n_hubs, n_neigh, seed)
    hub_set = set(hubs)
    in_deg_full = dict(graph.in_degree())
    out_deg_full = dict(graph.out_degree())

    # ── Build plotly figure ──────────────────────────────────────────────────
    fig = go.Figure()

    # Edge traces (non-hub edges, then hub edges)
    for u, v in sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_hub_edge = (u in hub_set or v in hub_set)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(
                color="#1e3a8a" if is_hub_edge else "#94a3b8",
                width=1.8 if is_hub_edge else 0.7,
            ),
            hoverinfo="none",
            showlegend=False,
        ))

    # Arrow annotations for direction (sample, otherwise too crowded)
    hub_edges = [(u, v) for u, v in sub.edges() if u in hub_set or v in hub_set]
    for u, v in hub_edges[:120]:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor="#1e3a8a",
        )

    # Ordinary nodes
    ord_nodes = [n for n in sub.nodes() if n not in hub_set]
    if ord_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ord_nodes],
            y=[pos[n][1] for n in ord_nodes],
            mode="markers",
            marker=dict(
                size=[8 + in_deg_full.get(n, 0) * 0.4 for n in ord_nodes],
                color="#5B9BD5",
                line=dict(width=1, color="white"),
            ),
            text=[
                f"User {n}<br>In-degree: {in_deg_full.get(n,0)}<br>"
                f"Out-degree: {out_deg_full.get(n,0)}"
                for n in ord_nodes
            ],
            hoverinfo="text",
            name="Ordinary users",
        ))

    # Hub nodes
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in hubs if n in pos],
        y=[pos[n][1] for n in hubs if n in pos],
        mode="markers+text",
        marker=dict(
            size=[22 + in_deg_full.get(n, 0) * 0.2 for n in hubs if n in pos],
            color="#1e3a8a",
            line=dict(width=2, color="white"),
        ),
        text=[f"U{n}" for n in hubs if n in pos],
        textposition="middle center",
        textfont=dict(color="white", size=9, family="Arial Bold"),
        customdata=[
            [n, in_deg_full.get(n, 0), out_deg_full.get(n, 0)]
            for n in hubs if n in pos
        ],
        hovertemplate=(
            "<b>User %{customdata[0]}</b><br>"
            "In-degree (votes received): %{customdata[1]}<br>"
            "Out-degree (votes cast): %{customdata[2]}<extra></extra>"
        ),
        name=f"Hub nodes (top {n_hubs})",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Wikipedia Vote Network — Top-{n_hubs} Hubs and Their Neighbours<br>"
                f"<sup>{sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges shown "
                f"(full graph: 7,115 nodes · 103,689 edges)</sup>"
            ),
            font=dict(size=14),
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        height=620,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Hub node stats table ────────────────────────────────────────────────
    st.subheader(f"Top-{n_hubs} Hub Node Statistics")
    hub_stats = pd.DataFrame([{
        "User ID":            n,
        "Votes Received (in-degree)":  in_deg_full.get(n, 0),
        "Votes Cast (out-degree)":     out_deg_full.get(n, 0),
        "Shown in subgraph": "✅" if n in sub else "❌",
    } for n in hubs])
    st.dataframe(hub_stats.set_index("User ID"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS & ALGORITHMS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results & Algorithms":
    st.title("Results & Algorithm Comparison")

    summary    = load_summary()
    per_source = load_per_source()
    topk       = load_topk()

    if summary is None:
        st.error("No results found. Run `python scripts/run_experiments.py --top-k 10` first.")
        st.stop()

    # ── Precision@K comparison ───────────────────────────────────────────────
    st.subheader("Mean Precision@10 — All Methods")

    methods_sorted = summary.sort_values("mean_precision_at_k", ascending=False)
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_bar = go.Figure()
        for _, row in methods_sorted.iterrows():
            m = row["method"]
            fig_bar.add_trace(go.Bar(
                x=[METHOD_LABELS.get(m, m)],
                y=[row["mean_precision_at_k"]],
                error_y=dict(type="data", array=[row["std_precision_at_k"]]),
                marker_color=METHOD_COLOURS.get(m, "#888"),
                name=METHOD_LABELS.get(m, m),
                hovertemplate=(
                    f"<b>{METHOD_LABELS.get(m, m)}</b><br>"
                    f"Mean P@10: {row['mean_precision_at_k']:.4f}<br>"
                    f"Std Dev: {row['std_precision_at_k']:.4f}<br>"
                    f"Total hits: {int(row['total_hits']):,}<extra></extra>"
                ),
            ))
        fig_bar.add_hline(y=0.091, line_dash="dot", line_color="#64748b",
                          annotation_text="Random baseline ≈ 0.091",
                          annotation_position="top right")
        fig_bar.update_layout(
            showlegend=False,
            yaxis_title="Mean Precision@10",
            yaxis_range=[0, 0.42],
            plot_bgcolor="#f8fafc",
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.markdown("#### Summary Table")
        disp = methods_sorted.copy()
        disp["Method"]   = disp["method"].map(lambda m: METHOD_LABELS.get(m, m))
        disp["P@10"]     = disp["mean_precision_at_k"].map("{:.4f}".format)
        disp["Std"]      = disp["std_precision_at_k"].map("{:.4f}".format)
        disp["Hits"]     = disp["total_hits"].astype(int).map("{:,}".format)
        disp["vs Random"] = [f"{v:.1f}×" for v in disp["mean_precision_at_k"] / 0.091]
        st.dataframe(
            disp[["Method", "P@10", "Std", "Hits", "vs Random"]].set_index("Method"),
            use_container_width=True,
        )
        st.caption(f"K = 10 · {int(methods_sorted['evaluated_sources'].iloc[0]):,} source nodes evaluated")

    st.markdown("---")

    # ── Per-source distribution ──────────────────────────────────────────────
    if per_source is not None:
        st.subheader("Per-Node Precision@10 Distribution")
        col_box, col_hist = st.columns(2)

        with col_box:
            fig_box = go.Figure()
            for m in ["personalized_pagerank", "common_neighbors", "adamic_adar", "jaccard"]:
                subset = per_source[per_source["method"] == m]["precision_at_k"]
                if subset.empty:
                    continue
                fig_box.add_trace(go.Box(
                    y=subset, name=METHOD_LABELS.get(m, m),
                    marker_color=METHOD_COLOURS.get(m, "#888"),
                    boxpoints="outliers", jitter=0.3,
                ))
            fig_box.update_layout(
                title="Precision@10 spread per source node",
                yaxis_title="Precision@10",
                showlegend=False,
                plot_bgcolor="#f8fafc",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col_hist:
            best_method = methods_sorted.iloc[0]["method"]
            ppr_data = per_source[per_source["method"] == best_method]["precision_at_k"]
            fig_hist = px.histogram(
                ppr_data, nbins=20,
                title=f"Distribution of P@10 — {METHOD_LABELS.get(best_method, best_method)}",
                labels={"value": "Precision@10", "count": "Number of nodes"},
                color_discrete_sequence=["#16a34a"],
            )
            fig_hist.update_layout(
                plot_bgcolor="#f8fafc",
                height=380,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Rank quality chart ───────────────────────────────────────────────────
    if topk is not None:
        st.subheader("Positive Edge Rate by Rank Position")
        st.markdown(
            "For each rank position (1–10), shows the fraction of predictions that are "
            "true positive test edges. Methods that push real edges to rank 1 learn better relevance."
        )
        rank_quality = (
            topk.groupby(["method", "rank"], as_index=False)["label"]
            .mean()
            .rename(columns={"label": "positive_rate"})
        )
        fig_rank = go.Figure()
        for m in ["personalized_pagerank", "common_neighbors", "adamic_adar", "jaccard"]:
            subset = rank_quality[rank_quality["method"] == m]
            fig_rank.add_trace(go.Scatter(
                x=subset["rank"], y=subset["positive_rate"],
                mode="lines+markers",
                name=METHOD_LABELS.get(m, m),
                line=dict(color=METHOD_COLOURS.get(m, "#888"), width=2),
                marker=dict(size=7),
            ))
        fig_rank.update_layout(
            xaxis_title="Rank position",
            yaxis_title="Fraction of true positive edges",
            plot_bgcolor="#f8fafc",
            height=380,
            legend=dict(x=0.7, y=0.98),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CROSS-DATASET COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Cross-Dataset":
    st.title("Cross-Dataset Comparison")
    st.markdown(
        "The same `GraphRAGPipeline` and the same algorithms run on two completely different "
        "graph types. **Personalised PageRank wins on both**, confirming that multi-hop graph "
        "traversal is the fundamental retrieval primitive for GraphRAG."
    )

    summary = load_summary()

    st.markdown("---")

    # ── Side-by-side bar charts ──────────────────────────────────────────────
    st.subheader("Algorithm Performance — Both Datasets")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Dataset 1 — Wikipedia Vote Network**  \n*Task: Link Prediction · Metric: Precision@10*")
        if summary is not None:
            fig_d1 = px.bar(
                summary.sort_values("mean_precision_at_k", ascending=False),
                x="method", y="mean_precision_at_k",
                error_y="std_precision_at_k",
                color="method", color_discrete_map=METHOD_COLOURS,
                labels={"method": "", "mean_precision_at_k": "Mean Precision@10"},
            )
            fig_d1.add_hline(y=0.091, line_dash="dot", line_color="gray",
                             annotation_text="Random ≈ 0.091")
            fig_d1.update_layout(showlegend=False, height=340,
                                 xaxis_ticktext=[METHOD_LABELS.get(m,"") for m in summary.sort_values("mean_precision_at_k",ascending=False)["method"]],
                                 xaxis_tickvals=list(summary.sort_values("mean_precision_at_k",ascending=False)["method"]))
            st.plotly_chart(fig_d1, use_container_width=True)

    with col_d2:
        st.markdown("**Dataset 2 — HotpotQA**  \n*Task: Context Retrieval · Metric: AUC-ROC*")
        fig_d2 = px.bar(
            HOTPOTQA_RESULTS, x="method", y="auc_roc",
            color="method", color_discrete_map=METHOD_COLOURS,
            labels={"method": "", "auc_roc": "AUC-ROC"},
        )
        fig_d2.add_hline(y=0.5, line_dash="dot", line_color="gray",
                         annotation_text="Random ≈ 0.500")
        fig_d2.update_layout(showlegend=False, height=340,
                             xaxis_ticktext=[METHOD_LABELS.get(m,"") for m in HOTPOTQA_RESULTS["method"]],
                             xaxis_tickvals=list(HOTPOTQA_RESULTS["method"]))
        st.plotly_chart(fig_d2, use_container_width=True)

    # ── Combined table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Combined Results Table")

    d1_dict = dict(zip(summary["method"], summary["mean_precision_at_k"])) if summary is not None else {}
    d2_dict = dict(zip(HOTPOTQA_RESULTS["method"], HOTPOTQA_RESULTS["auc_roc"]))

    all_methods = ["personalized_pagerank", "common_neighbors", "katz", "adamic_adar", "jaccard"]
    rows = []
    for m in all_methods:
        rows.append({
            "Method":           METHOD_LABELS.get(m, m),
            "D1 P@10":          f"{d1_dict[m]:.4f}" if m in d1_dict else "—",
            "D1 vs Random":     f"{d1_dict[m]/0.091:.1f}×" if m in d1_dict else "—",
            "D2 AUC-ROC":       f"{d2_dict[m]:.4f}" if m in d2_dict else "—",
            "D2 vs Random":     f"+{(d2_dict[m]-0.5)/0.5*100:.0f}%" if m in d2_dict else "—",
        })
    combined_df = pd.DataFrame(rows).set_index("Method")
    st.dataframe(combined_df, use_container_width=True)

    # ── Key findings ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Findings")
    f1, f2, f3, f4 = st.columns(4)
    f1.success("🏆 **PPR wins on both datasets**  \nP@10=0.337 · AUC=0.808")
    f2.error("🔴 **Jaccard lowest on both**  \nP@10=0.294 · AUC=0.630")
    f3.warning("⚡ **Katz strong on D2**  \nAUC=0.725 — 2nd best on HotpotQA")
    f4.info("🔌 **One pipeline, two datasets**  \nSame interface, auto metric selection")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTION EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediction Explorer":
    st.title("Prediction Explorer")
    st.markdown(
        "Look up the Top-10 predicted links for any source node and compare "
        "what each algorithm recommends. Green rows are true positive test edges."
    )

    topk = load_topk()
    if topk is None:
        st.error("No predictions found. Run `python scripts/run_experiments.py --top-k 10` first.")
        st.stop()

    graph = load_graph()
    in_deg_full  = dict(graph.in_degree())  if graph else {}
    out_deg_full = dict(graph.out_degree()) if graph else {}

    available_nodes = sorted(topk["source"].unique().tolist())
    st.caption(f"{len(available_nodes):,} source nodes have predictions.")

    col_input, col_rand = st.columns([3, 1])
    with col_rand:
        if st.button("🎲 Random node"):
            st.session_state["selected_node"] = random.choice(available_nodes)
    default_node = st.session_state.get("selected_node", available_nodes[0])
    selected = col_input.selectbox(
        "Select source node (user ID)",
        options=available_nodes,
        index=available_nodes.index(default_node) if default_node in available_nodes else 0,
    )
    st.session_state["selected_node"] = selected

    # ── Node info card ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("User ID", selected)
    c2.metric("Votes received (in-degree)", in_deg_full.get(selected, "—"))
    c3.metric("Votes cast (out-degree)", out_deg_full.get(selected, "—"))

    node_preds = topk[topk["source"] == selected].copy()
    node_preds["Method"] = node_preds["method"].map(lambda m: METHOD_LABELS.get(m, m))
    node_preds["Hit ✓"] = node_preds["label"].map(lambda l: "✅ True positive" if l == 1 else "")

    st.markdown("---")

    # ── Side-by-side per method ──────────────────────────────────────────────
    st.subheader("Top-10 Predictions per Algorithm")
    cols = st.columns(4)
    for i, m in enumerate(["personalized_pagerank", "common_neighbors", "adamic_adar", "jaccard"]):
        subset = node_preds[node_preds["method"] == m][["rank", "target", "score", "label"]].copy()
        subset["score"] = subset["score"].map("{:.6f}".format)
        subset = subset.rename(columns={
            "rank": "Rank", "target": "Target node",
            "score": "Score", "label": "True link",
        })
        subset["True link"] = subset["True link"].map(lambda l: "✅" if l == 1 else "")
        hits = (node_preds[node_preds["method"] == m]["label"] == 1).sum()
        cols[i].markdown(
            f"**{METHOD_LABELS.get(m, m)}**  \n"
            f"<span style='color:{'#16a34a' if hits > 0 else '#dc2626'}'>"
            f"{hits}/10 hits</span>",
            unsafe_allow_html=True,
        )
        cols[i].dataframe(
            subset.set_index("Rank"),
            use_container_width=True,
            height=380,
        )

    # ── Score comparison across methods ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Score Comparison Across Methods")
    st.markdown("For each target node predicted by any algorithm, compare the raw score each method assigned.")

    scored = load_topk()
    if scored is not None:
        pivot_data = []
        for m in ["personalized_pagerank", "common_neighbors", "adamic_adar", "jaccard"]:
            sub = scored[(scored["source"] == selected) & (scored["method"] == m)][["target", "score", "label"]]
            sub = sub.rename(columns={"score": METHOD_LABELS.get(m, m)})
            pivot_data.append(sub.set_index(["target", "label"]))

        if pivot_data:
            try:
                combined = pivot_data[0].join(pivot_data[1:], how="outer").reset_index()
                combined["True link"] = combined["label"].map(lambda l: "✅" if l == 1 else "")
                combined = combined.drop(columns=["label"])
                for col in list(METHOD_LABELS.values()):
                    if col in combined.columns:
                        combined[col] = combined[col].map(
                            lambda x: f"{x:.6f}" if pd.notna(x) else "—"
                        )
                st.dataframe(
                    combined.set_index("target"),
                    use_container_width=True,
                    height=380,
                )
            except Exception:
                st.info("Could not pivot scores — methods predict different target sets.")
