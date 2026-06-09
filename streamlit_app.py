
# -*- coding: utf-8 -*-
"""
K-평균 군집화를 활용한 수학 학습자 유형 분석 Streamlit 앱
- 업로드 파일 기반 자동 분석
- 구쌤기말.ipynb 분석 흐름 반영
- math_cluster_dashboard.html의 보고서형 화면 구성 반영
"""

from __future__ import annotations

import base64
import io
import re
import textwrap
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="수학 학습자 유형 분석 보고서",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
  --bg:#f4f5f7;
  --white:#ffffff;
  --ink:#111827;
  --ink2:#374151;
  --muted:#6b7280;
  --line:#e5e7eb;
  --black:#0b0f19;
  --red:#e11d48;
  --red-soft:#fff1f2;
  --blue:#2563eb;
  --blue-soft:#eff6ff;
  --green:#059669;
  --green-soft:#ecfdf5;
  --amber:#d97706;
  --amber-soft:#fffbeb;
  --purple:#7c3aed;
  --purple-soft:#f5f3ff;
  --shadow:0 24px 60px rgba(15,23,42,.08);
}
html, body, [class*="css"] {
  font-family: Helvetica, "Helvetica Neue", Arial, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
  color: var(--ink);
}
body, .stApp { background: var(--bg); }
.block-container {
  padding-top: 1.1rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}
.report-header {
  position: relative;
  overflow: hidden;
  min-height: 260px;
  background:
    radial-gradient(circle at 78% 18%, rgba(225,29,72,.28), transparent 30%),
    linear-gradient(135deg, #0b0f19 0%, #141b2d 50%, #20293c 100%);
  border: 0;
  border-radius: 30px;
  padding: 42px 46px;
  margin-bottom: 22px;
  box-shadow: var(--shadow);
}
.report-header::after {
  content: "K";
  position: absolute;
  right: 42px;
  bottom: -78px;
  font-size: 300px;
  line-height: .8;
  font-weight: 900;
  color: rgba(255,255,255,.08);
  letter-spacing: -18px;
}
.kicker {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,.10);
  color: #f8fafc;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  margin-bottom: 20px;
}
.kicker::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--red); }
.report-title {
  max-width: 760px;
  font-size: clamp(34px, 4.5vw, 58px);
  font-weight: 900;
  color: #ffffff;
  letter-spacing: -2.4px;
  line-height: 1.06;
  margin-bottom: 14px;
}
.report-subtitle {
  max-width: 760px;
  font-size: 16px;
  color: rgba(255,255,255,.76);
  line-height: 1.75;
}
.meta-wrap {
  display:flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 26px;
  position:relative;
  z-index:1;
}
.meta-chip {
  display:inline-flex;
  align-items:center;
  gap: 8px;
  background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.16);
  border-radius:999px;
  padding:8px 13px;
  color:#f8fafc;
  font-size:12px;
  backdrop-filter: blur(8px);
}
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.soft-card, .factor-card, .cluster-card {
  background: rgba(255,255,255,.96);
  border: 1px solid rgba(229,231,235,.95);
  border-radius: 24px;
  box-shadow: 0 14px 34px rgba(15,23,42,.055);
}
.soft-card {
  padding: 26px 28px;
  margin-bottom: 18px;
}
.small-label {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 2px;
  color: var(--red);
  text-transform: uppercase;
  margin-bottom: 10px;
}
.card-title {
  font-size: 24px;
  font-weight: 900;
  color: var(--ink);
  letter-spacing: -0.9px;
  line-height: 1.22;
  margin-bottom: 8px;
}
.card-text {
  font-size: 15px;
  color: var(--ink2);
  line-height: 1.75;
}
.factor-card {
  padding:22px;
  height:100%;
  transition: transform .15s ease, box-shadow .15s ease;
}
.factor-card:hover { transform: translateY(-2px); box-shadow: 0 20px 42px rgba(15,23,42,.09); }
.factor-icon {
  width:44px;
  height:44px;
  border-radius:14px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:900;
  margin-bottom:14px;
}
.cluster-card {
  padding:24px;
  height:100%;
  position:relative;
  overflow:hidden;
}
.cluster-stripe {
  position:absolute;
  top:0;
  left:0;
  right:0;
  height:6px;
}
.cluster-name {
  font-size:28px;
  font-weight:900;
  letter-spacing:-1.1px;
  margin:12px 0 6px 0;
}
.cluster-sub {
  font-size:14px;
  color:var(--ink2);
  line-height:1.65;
  margin-bottom:14px;
}
.badge {
  display:inline-block;
  border-radius:999px;
  padding:6px 11px;
  font-size:12px;
  font-weight:800;
  margin: 3px 4px 3px 0;
}
.badge-red { background:var(--red-soft); color:var(--red); }
.badge-green { background:var(--green-soft); color:var(--green); }
.badge-amber { background:var(--amber-soft); color:var(--amber); }
.badge-blue { background:var(--blue-soft); color:var(--blue); }
.callout, .warning-callout {
  border:0;
  border-radius:24px;
  padding:20px 24px;
  line-height:1.75;
  margin:14px 0 20px 0;
  box-shadow: 0 12px 30px rgba(15,23,42,.05);
}
.callout {
  background:linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
  color:var(--ink2);
  border-left:6px solid var(--blue);
}
.warning-callout {
  background:linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
  color:var(--ink2);
  border-left:6px solid var(--amber);
}
.slide-hero {
  background:#0b0f19;
  color:#fff;
  border-radius:30px;
  padding:40px;
  margin:18px 0;
  box-shadow:var(--shadow);
}
.slide-hero h2 { font-size:44px; line-height:1.08; letter-spacing:-1.8px; margin:0 0 12px 0; }
.slide-hero p { color:rgba(255,255,255,.76); font-size:16px; line-height:1.7; margin:0; }
.slide-number {
  font-size:72px;
  font-weight:900;
  letter-spacing:-3px;
  color:var(--red);
}
.slide-note {
  background:#fff;
  border:1px solid var(--line);
  border-radius:22px;
  padding:18px 20px;
  font-size:14px;
  color:var(--muted);
  line-height:1.65;
  box-shadow:0 10px 24px rgba(15,23,42,.04);
}
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px;
  box-shadow: 0 10px 30px rgba(15,23,42,.04);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 11px 16px;
  background: transparent;
  border: 0;
  font-weight: 800;
  color: var(--muted);
}
.stTabs [aria-selected="true"] {
  background:#0b0f19 !important;
  color:#ffffff !important;
  border:0 !important;
}
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px 18px;
  box-shadow:0 14px 34px rgba(15,23,42,.055);
}
div[data-testid="stMetric"] label { color:var(--muted); font-weight:800; }
div[data-testid="stMetricValue"] { font-weight:900; letter-spacing:-.7px; }
.stDataFrame, div[data-testid="stPlotlyChart"] {
  border-radius: 22px;
  overflow:hidden;
}
section[data-testid="stSidebar"] {
  background:#ffffff;
  border-right:1px solid var(--line);
}
hr { margin: 1.3rem 0; border-color:var(--line); }
</style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FACTOR_INFO = {
    "math_anxiety_mean": {
        "name": "수학불안",
        "short": "A",
        "prefix": "A",
        "color": "#e03131",
        "bg": "#fff0f0",
        "desc": "수학 상황에서 느끼는 긴장, 걱정, 회피 반응",
        "positive_high": False,
    },
    "math_self_efficacy_mean": {
        "name": "자기효능감",
        "short": "E",
        "prefix": "E",
        "color": "#1b64da",
        "bg": "#eef3fd",
        "desc": "수학 과제를 해결할 수 있다는 믿음과 자신감",
        "positive_high": True,
    },
    "math_interest_mean": {
        "name": "수학흥미",
        "short": "I",
        "prefix": "I",
        "color": "#0d9e75",
        "bg": "#e6f7f1",
        "desc": "수학 학습에 대한 흥미, 성취감, 참여 의지",
        "positive_high": True,
    },
    "learning_attitude_mean": {
        "name": "학습태도",
        "short": "T",
        "prefix": "T",
        "color": "#c47d0e",
        "bg": "#fff8e6",
        "desc": "계획, 성실성, 질문, 협력 등 학습 행동 습관",
        "positive_high": True,
    },
}
CLUSTER_VARS = list(FACTOR_INFO.keys())
VAR_LABELS = {k: v["name"] for k, v in FACTOR_INFO.items()}
REVERSE_ITEMS = ["T6"]


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def section(label: str, title: str, text: str = ""):
    st.markdown(
        f"""
<div class="soft-card">
  <div class="small-label">{label}</div>
  <div class="card-title">{title}</div>
  <div class="card-text">{text}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def report_header(n: Optional[int] = None, k: Optional[int] = None):
    n_txt = f"{n:,}명 분석" if n is not None else "파일 업로드 후 자동 분석"
    k_txt = f"K = {k} 군집" if k is not None else "K 자동 탐색"
    st.markdown(
        f"""
<div class="report-header">
  <div class="kicker">K-Means Clustering Analysis</div>
  <div class="report-title">수학 학습자 유형 분석 보고서</div>
  <div class="report-subtitle">
    수학불안 · 자기효능감 · 수학흥미 · 학습태도 설문 데이터를 바탕으로 학생 집단을 자동 유형화하고,
    분석 결과를 수업 지원 방안까지 연결하는 보고서형 웹앱입니다.
  </div>
  <div class="meta-wrap">
    <span class="meta-chip"><span class="dot" style="background:#1b64da"></span>{k_txt}</span>
    <span class="meta-chip"><span class="dot" style="background:#0d9e75"></span>4개 하위요인</span>
    <span class="meta-chip"><span class="dot" style="background:#c47d0e"></span>5점 리커트 척도</span>
    <span class="meta-chip"><span class="dot" style="background:#6343c8"></span>표준화 Z-score</span>
    <span class="meta-chip"><span class="dot" style="background:#e03131"></span>{n_txt}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    """Convert common Likert values to numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip()
    # Extract the first numeric token, so strings such as "5. 매우 그렇다" work.
    extracted = text.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="cp949")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded, sheet_name=0)
    raise ValueError("엑셀(.xlsx, .xls) 또는 CSV 파일만 업로드할 수 있습니다.")


def detect_items(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = [str(c).strip() for c in df.columns]
    mapping = {}
    for key, info in FACTOR_INFO.items():
        prefix = info["prefix"]
        pat = re.compile(rf"^\s*{prefix}\s*\d+\s*[\.\)]?", re.IGNORECASE)
        found = [c for c in cols if pat.search(c)]
        # stable sort by item number
        def item_no(col):
            m = re.search(rf"{prefix}\s*(\d+)", col, re.IGNORECASE)
            return int(m.group(1)) if m else 999
        mapping[key] = sorted(found, key=item_no)
    return mapping


def get_item_code(col: str) -> str:
    m = re.search(r"\b([AEIT]\s*\d+)\b", str(col), flags=re.IGNORECASE)
    return m.group(1).upper().replace(" ", "") if m else str(col)


def compute_factors(df: pd.DataFrame, item_map: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    out = df.copy()
    used_cols = []
    notes = []
    for factor, cols in item_map.items():
        if not cols:
            out[factor] = np.nan
            notes.append(f"{VAR_LABELS[factor]} 문항을 찾지 못했습니다.")
            continue
        num_df = pd.DataFrame(index=out.index)
        for col in cols:
            vals = safe_numeric(out[col])
            code = get_item_code(col)
            if code in REVERSE_ITEMS:
                # 5점 리커트 기준 역채점: 1↔5, 2↔4, 3↔3
                vals = 6 - vals
            num_df[col] = vals
        out[factor] = num_df.mean(axis=1)
        used_cols.extend(cols)
    factor_df = out[CLUSTER_VARS].copy()
    return out, factor_df, notes


def get_s1_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if str(c).strip().upper().startswith("S1")]
    if candidates:
        return candidates[0]
    candidates = [c for c in df.columns if "어렵" in str(c) and "수학" in str(c)]
    return candidates[0] if candidates else None


@dataclass
class AnalysisResult:
    df_raw: pd.DataFrame
    df_factor: pd.DataFrame
    df_clean: pd.DataFrame
    item_map: Dict[str, List[str]]
    selected_vars: List[str]
    scaler: StandardScaler
    X_scaled: np.ndarray
    k: int
    k_recommended: Optional[int]
    elbow_df: pd.DataFrame
    silhouette_df: pd.DataFrame
    result_df: pd.DataFrame
    profile_raw: pd.DataFrame
    profile_scaled: pd.DataFrame
    pca_df: pd.DataFrame
    pca_model: PCA
    notes: List[str]


def run_analysis(df: pd.DataFrame, selected_vars: List[str], k: Optional[int], random_state: int = 42) -> AnalysisResult:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    item_map = detect_items(df)
    df_factor_all, df_factor, notes = compute_factors(df, item_map)

    selected_vars = [v for v in selected_vars if v in df_factor.columns]
    df_clean = df_factor_all.dropna(subset=selected_vars).copy()

    if len(df_clean) < 3:
        raise ValueError("분석 가능한 응답이 너무 적습니다. 선택한 문항/변수의 결측치를 확인해주세요.")

    X = df_clean[selected_vars].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow & silhouette
    max_k = int(min(8, len(df_clean) - 1))
    rows_elbow, rows_sil = [], []
    for kk in range(1, max_k + 1):
        km = KMeans(n_clusters=kk, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)
        rows_elbow.append({"K": kk, "Inertia": km.inertia_})
        if kk >= 2:
            try:
                sil = silhouette_score(X_scaled, labels)
            except Exception:
                sil = np.nan
            rows_sil.append({"K": kk, "Silhouette": sil})
    elbow_df = pd.DataFrame(rows_elbow)
    silhouette_df = pd.DataFrame(rows_sil)
    k_recommended = None
    if not silhouette_df.empty and silhouette_df["Silhouette"].notna().any():
        k_recommended = int(silhouette_df.loc[silhouette_df["Silhouette"].idxmax(), "K"])

    if k is None:
        k_final = k_recommended or min(2, max_k)
    else:
        k_final = int(k)
    k_final = max(2, min(k_final, max_k))

    model = KMeans(n_clusters=k_final, random_state=random_state, n_init=30)
    labels0 = model.fit_predict(X_scaled)

    # Label clusters pedagogically: sort by positive learning index
    profile_tmp = pd.DataFrame(X_scaled, columns=selected_vars).assign(cluster0=labels0).groupby("cluster0")[selected_vars].mean()
    def score_row(row):
        anxiety = row.get("math_anxiety_mean", 0)
        positives = [row.get(v, 0) for v in selected_vars if v != "math_anxiety_mean"]
        return (np.mean(positives) if positives else 0) - anxiety
    sorted_clusters = sorted(profile_tmp.index, key=lambda c: score_row(profile_tmp.loc[c]), reverse=True)
    remap = {old: new + 1 for new, old in enumerate(sorted_clusters)}
    labels = np.array([remap[x] for x in labels0])

    result_df = df_clean.copy()
    result_df["cluster"] = labels
    result_df["cluster_label"] = result_df["cluster"].map(lambda x: f"군집 {x}")

    profile_raw = result_df.groupby("cluster")[selected_vars].mean()
    scaled_df = pd.DataFrame(X_scaled, columns=selected_vars, index=df_clean.index)
    scaled_df["cluster"] = labels
    profile_scaled = scaled_df.groupby("cluster")[selected_vars].mean()

    pca_model = PCA(n_components=2, random_state=random_state)
    pcs = pca_model.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "cluster": labels.astype(str),
        "cluster_num": labels,
    }, index=df_clean.index)
    for v in selected_vars:
        pca_df[VAR_LABELS[v]] = df_clean[v].values
    s1_col = get_s1_col(df)
    if s1_col:
        pca_df["S1"] = df.loc[df_clean.index, s1_col].astype(str).replace("nan", "")

    return AnalysisResult(
        df_raw=df,
        df_factor=df_factor_all,
        df_clean=df_clean,
        item_map=item_map,
        selected_vars=selected_vars,
        scaler=scaler,
        X_scaled=X_scaled,
        k=k_final,
        k_recommended=k_recommended,
        elbow_df=elbow_df,
        silhouette_df=silhouette_df,
        result_df=result_df,
        profile_raw=profile_raw,
        profile_scaled=profile_scaled,
        pca_df=pca_df,
        pca_model=pca_model,
        notes=notes,
    )


def interpret_cluster(profile_scaled: pd.Series, profile_raw: pd.Series, selected_vars: List[str]) -> Dict[str, str]:
    anxiety = profile_scaled.get("math_anxiety_mean", np.nan)
    positives = [profile_scaled.get(v, np.nan) for v in selected_vars if v != "math_anxiety_mean"]
    pos_mean = np.nanmean(positives) if positives else np.nan

    if not np.isnan(anxiety) and not np.isnan(pos_mean):
        if anxiety < -0.15 and pos_mean > 0.15:
            title = "안정·성장형"
            subtitle = "수학불안은 낮고 자기효능감·흥미·학습태도가 높은 집단"
            strategy = "심화 도전 과제, 탐구 활동, 또래 설명·멘토링 역할을 통해 잠재력을 확장하는 전략이 적합합니다."
            color = "#0d9e75"
        elif anxiety > 0.15 and pos_mean < -0.15:
            title = "불안·지원필요형"
            subtitle = "수학불안은 높고 긍정적 학습 특성은 낮은 집단"
            strategy = "정서적 안정, 작은 성공 경험, 단계적 난이도 조절, 즉각적 피드백을 우선 제공하는 전략이 필요합니다."
            color = "#e03131"
        elif anxiety > 0.15 and pos_mean > 0.15:
            title = "긴장·성취추구형"
            subtitle = "학습 의지는 있으나 수학 상황에서 긴장도 함께 높은 집단"
            strategy = "도전 과제를 제공하되 평가 부담을 낮추고, 풀이 과정 중심 피드백과 시험불안 완화 전략을 함께 지원합니다."
            color = "#6343c8"
        else:
            title = "보통·잠재형"
            subtitle = "전체 평균에 가까운 중간 집단으로, 계기에 따라 성장 방향이 달라질 수 있는 집단"
            strategy = "생활·전공 연계 과제, 선택권 있는 활동, 소그룹 협력으로 흥미와 자기효능감을 깨우는 전략이 적합합니다."
            color = "#c47d0e"
    else:
        title = "혼합형"
        subtitle = "선택 변수 기준에서 복합적인 특성을 보이는 집단"
        strategy = "개별 학생의 강점과 어려움을 함께 확인하며 맞춤 피드백을 제공합니다."
        color = "#1b64da"

    # dominant features
    strengths, needs = [], []
    for var in selected_vars:
        z = profile_scaled.get(var, np.nan)
        nm = VAR_LABELS[var]
        if var == "math_anxiety_mean":
            if z < -0.3:
                strengths.append("낮은 수학불안")
            elif z > 0.3:
                needs.append("수학불안 완화")
        else:
            if z > 0.3:
                strengths.append(f"높은 {nm}")
            elif z < -0.3:
                needs.append(f"{nm} 지원")

    return {
        "title": title,
        "subtitle": subtitle,
        "strategy": strategy,
        "color": color,
        "strengths": ", ".join(strengths) if strengths else "뚜렷한 강점은 평균 수준",
        "needs": ", ".join(needs) if needs else "긴급 지원 요인은 크지 않음",
    }


# ─────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────
def fig_elbow(res: AnalysisResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.elbow_df["K"], y=res.elbow_df["Inertia"],
        mode="lines+markers+text",
        text=[f"K={int(k)}" for k in res.elbow_df["K"]],
        textposition="top center",
        line=dict(width=3, color="#1b64da"),
        marker=dict(size=9, color="#1b64da"),
        name="Inertia",
    ))
    fig.add_vline(x=res.k, line_dash="dash", line_color="#e03131", annotation_text=f"선택 K={res.k}")
    fig.update_layout(
        title="Elbow Method: K 후보 탐색",
        xaxis_title="군집 수 K",
        yaxis_title="Inertia",
        height=420,
        margin=dict(l=20, r=20, t=60, b=30),
        template="plotly_white",
    )
    return fig


def fig_silhouette(res: AnalysisResult) -> go.Figure:
    df = res.silhouette_df.copy()
    df["K"] = df["K"].astype(str)
    fig = px.bar(
        df, x="K", y="Silhouette", text=df["Silhouette"].round(3),
        title="Silhouette Score: 군집 분리도 확인",
        color_discrete_sequence=["#0d9e75"],
    )
    fig.add_hline(y=df["Silhouette"].max(), line_dash="dot", line_color="#1b64da")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="군집 수 K",
        yaxis_title="실루엣 점수",
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


def fig_cluster_counts(res: AnalysisResult) -> go.Figure:
    counts = res.result_df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    counts["군집"] = counts["cluster"].map(lambda x: f"군집 {x}")
    colors = ["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"][: len(counts)]
    fig = px.bar(
        counts, x="군집", y="count", text="count",
        title="군집별 학생 수",
        color="군집",
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_title="학생 수",
        xaxis_title="",
        height=380,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


def fig_heatmap(profile: pd.DataFrame, title: str, zscore: bool) -> go.Figure:
    labels = [VAR_LABELS.get(c, c) for c in profile.columns]
    y = [f"군집 {i}" for i in profile.index]
    vals = profile.values
    colorscale = "RdYlGn" if zscore else "Blues"
    zmid = 0 if zscore else None
    text = np.vectorize(lambda x: f"{x:+.2f}" if zscore else f"{x:.2f}")(vals)
    fig = go.Figure(data=go.Heatmap(
        z=vals, x=labels, y=y, text=text, texttemplate="%{text}",
        colorscale=colorscale, zmid=zmid, colorbar=dict(title="Z" if zscore else "평균"),
        hovertemplate="군집=%{y}<br>요인=%{x}<br>값=%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=350, template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


def fig_radar(res: AnalysisResult) -> go.Figure:
    categories = [VAR_LABELS[c] for c in res.profile_scaled.columns]
    categories_closed = categories + [categories[0]]
    fig = go.Figure()
    palette = ["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"]
    for idx, cluster in enumerate(res.profile_scaled.index):
        values = res.profile_scaled.loc[cluster].tolist()
        values_closed = values + [values[0]]
        interp = interpret_cluster(res.profile_scaled.loc[cluster], res.profile_raw.loc[cluster], res.selected_vars)
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=f"군집 {cluster} · {interp['title']}",
            line=dict(color=palette[idx % len(palette)], width=2),
            opacity=0.72,
        ))
    fig.update_layout(
        title="표준화 군집 프로파일 레이더 차트",
        polar=dict(radialaxis=dict(visible=True, range=[min(-2, float(res.profile_scaled.min().min()) - .2), max(2, float(res.profile_scaled.max().max()) + .2)])),
        height=520,
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def fig_pca(res: AnalysisResult) -> go.Figure:
    df = res.pca_df.copy()
    df["군집"] = df["cluster"].map(lambda x: f"군집 {x}")
    hover_cols = [VAR_LABELS[v] for v in res.selected_vars if VAR_LABELS[v] in df.columns]
    if "S1" in df.columns:
        hover_cols.append("S1")
    fig = px.scatter(
        df, x="PC1", y="PC2", color="군집",
        hover_data=hover_cols,
        title=f"PCA 2차원 군집 분포 · 설명분산 {res.pca_model.explained_variance_ratio_[0]*100:.1f}% + {res.pca_model.explained_variance_ratio_[1]*100:.1f}%",
        color_discrete_sequence=["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"],
        opacity=0.78,
    )
    centers = df.groupby("군집")[["PC1", "PC2"]].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=centers["PC1"], y=centers["PC2"],
        mode="markers+text",
        text=centers["군집"],
        textposition="top center",
        marker=dict(symbol="x", size=18, color="#191f28", line=dict(width=3)),
        name="Centroid",
    ))
    fig.update_layout(
        height=520, template="plotly_white",
        margin=dict(l=20, r=20, t=70, b=30),
    )
    return fig


def fig_anxiety_attitude(res: AnalysisResult) -> Optional[go.Figure]:
    if "math_anxiety_mean" not in res.selected_vars or "learning_attitude_mean" not in res.selected_vars:
        return None
    df = res.result_df[["math_anxiety_mean", "learning_attitude_mean", "cluster"]].dropna().copy()
    if len(df) < 3:
        return None
    df["군집"] = df["cluster"].map(lambda x: f"군집 {x}")
    fig = px.scatter(
        df, x="math_anxiety_mean", y="learning_attitude_mean",
        color="군집",
        trendline="ols",
        title="수학불안과 학습태도의 관계",
        labels={"math_anxiety_mean": "수학불안 평균", "learning_attitude_mean": "학습태도 평균"},
        color_discrete_sequence=["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"],
        opacity=0.76,
    )
    fig.update_layout(height=500, template="plotly_white", margin=dict(l=20, r=20, t=60, b=30))
    return fig


def make_centroid_iterations(X2: np.ndarray, k: int, random_state: int = 42, steps: int = 5):
    rng = np.random.default_rng(random_state)
    if len(X2) <= k:
        return []
    idx = rng.choice(len(X2), size=k, replace=False)
    centers = X2[idx].copy()
    frames = []
    labels = np.zeros(len(X2), dtype=int)
    for step in range(steps):
        d = ((X2[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        frames.append((step, labels.copy(), centers.copy()))
        new_centers = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                new_centers[j] = X2[labels == j].mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            frames.append((step + 1, labels.copy(), centers.copy()))
            break
        centers = new_centers
    return frames


def fig_centroid_step(res: AnalysisResult, step_index: int) -> go.Figure:
    X2 = res.pca_df[["PC1", "PC2"]].values
    frames = make_centroid_iterations(X2, res.k, steps=6)
    if not frames:
        return go.Figure()
    step, labels, centers = frames[min(step_index, len(frames)-1)]
    df = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1], "군집": [f"C{x+1}" for x in labels]})
    fig = px.scatter(
        df, x="PC1", y="PC2", color="군집",
        color_discrete_sequence=["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"],
        title=f"K-평균 중심 이동 시뮬레이션 · Step {step}",
        opacity=0.62,
    )
    fig.add_trace(go.Scatter(
        x=centers[:, 0], y=centers[:, 1],
        mode="markers+text",
        text=[f"중심 {i+1}" for i in range(len(centers))],
        textposition="top center",
        marker=dict(symbol="x", size=18, color="#191f28", line=dict(width=3)),
        name="Centroid",
    ))
    fig.update_layout(height=380, template="plotly_white", margin=dict(l=20, r=20, t=60, b=30))
    return fig


def s1_frequency_table(res: AnalysisResult) -> Optional[pd.DataFrame]:
    s1_col = get_s1_col(res.df_raw)
    if not s1_col or s1_col not in res.df_raw.columns:
        return None
    temp = res.result_df[["cluster"]].copy()
    temp["S1"] = res.df_raw.loc[res.result_df.index, s1_col].astype(str)
    temp = temp[temp["S1"].str.strip().ne("") & temp["S1"].ne("nan")]
    if temp.empty:
        return None

    keywords = {
        "문제 난이도": ["어려", "심화", "응용", "문제"],
        "개념 이해": ["개념", "이해", "공식", "원리"],
        "계산/풀이": ["계산", "풀이", "풀", "식"],
        "기억/암기": ["기억", "암기", "외우"],
        "시간/시험": ["시간", "시험", "평가"],
        "집중/태도": ["집중", "귀찮", "포기", "꾸준"],
        "질문/설명": ["질문", "설명", "선생", "친구"],
    }
    rows = []
    for cluster, group in temp.groupby("cluster"):
        texts = " ".join(group["S1"].tolist())
        for cat, pats in keywords.items():
            cnt = sum(texts.count(p) for p in pats)
            rows.append({"cluster": f"군집 {cluster}", "category": cat, "count": cnt})
    freq = pd.DataFrame(rows)
    return freq


def fig_s1_keywords(res: AnalysisResult) -> Optional[go.Figure]:
    freq = s1_frequency_table(res)
    if freq is None or freq["count"].sum() == 0:
        return None
    fig = px.bar(
        freq, x="category", y="count", color="cluster", barmode="group",
        title="S1 자유응답 핵심어 분포",
        labels={"category": "어려움 범주", "count": "언급 횟수", "cluster": "군집"},
        color_discrete_sequence=["#0d9e75", "#e03131", "#c47d0e", "#6343c8", "#1b64da", "#8b95a1"],
    )
    fig.update_layout(height=420, template="plotly_white", margin=dict(l=20, r=20, t=60, b=30))
    return fig


def report_markdown(res: AnalysisResult) -> str:
    lines = []
    lines.append("# K-평균 군집화를 활용한 수학 학습자 유형 분석 보고서\n")
    lines.append("## 1. 분석 개요")
    lines.append(f"- 전체 응답 수: {len(res.df_raw):,}명")
    lines.append(f"- 분석 가능 응답 수: {len(res.df_clean):,}명")
    lines.append(f"- 최종 선택 K: {res.k}")
    if res.k_recommended:
        lines.append(f"- 실루엣 기준 추천 K: {res.k_recommended}")
    lines.append("- 분석 변수: " + ", ".join(VAR_LABELS[v] for v in res.selected_vars))
    lines.append("\n## 2. 군집별 특성")
    for cluster in res.profile_scaled.index:
        interp = interpret_cluster(res.profile_scaled.loc[cluster], res.profile_raw.loc[cluster], res.selected_vars)
        lines.append(f"\n### 군집 {cluster}: {interp['title']}")
        lines.append(f"- 특징: {interp['subtitle']}")
        lines.append(f"- 강점: {interp['strengths']}")
        lines.append(f"- 지원 필요: {interp['needs']}")
        lines.append(f"- 지도 방안: {interp['strategy']}")
        lines.append("- 표준화 평균")
        for var in res.selected_vars:
            lines.append(f"  - {VAR_LABELS[var]}: {res.profile_scaled.loc[cluster, var]:+.2f}")
    lines.append("\n## 3. 종합 시사점")
    lines.append("학생 군집은 성적 자체가 아니라 정서·동기·행동 요인을 중심으로 해석해야 합니다. 따라서 군집명은 학생을 고정적으로 분류하기 위한 이름이 아니라, 현재 수업 지원 방향을 찾기 위한 탐색적 진단명으로 활용하는 것이 바람직합니다.")
    return "\n".join(lines)




# ─────────────────────────────────────────────────────────────
# K 탐색 결과 리포트 화면
# ─────────────────────────────────────────────────────────────
def display_pdf_file(pdf_path: str, height: int = 640):
    # Streamlit 화면 안에 PDF를 표시합니다. PDF 파일은 app.py와 같은 폴더에 있어야 합니다.
    path = Path(pdf_path)
    if not path.exists():
        st.info("원본 PDF 파일을 찾을 수 없습니다. 앱 폴더에 PDF 파일을 함께 넣으면 원본 리포트를 볼 수 있습니다.")
        return
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="{height}"
        style="border:1px solid #e8eaed; border-radius:14px; background:#fff;"
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def render_k_exploration_result_tab():
    # 업로드한 PDF 내용을 웹앱 탭 안에서 요약·표·원본 PDF로 보여줍니다.
    section(
        "K Exploration Result",
        "변수 선택이 학생 군집화에 미치는 영향",
        "변수 조합에 따라 최적 K는 달라지지만, 수업 적용 관점에서는 4개 변수를 모두 포함한 K=2 모델이 가장 실용적입니다.",
    )

    st.markdown(
        """
<div class="callout">
  <b>핵심 결론</b><br>
  수학불안, 자기효능감, 수학흥미, 학습태도를 모두 포함하면 <b>K=2</b>가 가장 안정적입니다.
  학생의 마음은 다양하지만, 수업 지원 관점에서는 두 집단으로 보는 것이 가장 실용적입니다.
</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="factor-card"><b>종합성</b><br><span style="color:#4e5968;font-size:13px">4개 변수가 학생의 정서·동기·행동을 함께 반영합니다.</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="factor-card"><b>해석 가능성</b><br><span style="color:#4e5968;font-size:13px">K=2는 결과가 단순하여 학생 유형 해석이 쉽습니다.</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="factor-card"><b>수업 적용성</b><br><span style="color:#4e5968;font-size:13px">두 집단으로 나누면 맞춤형 지원 전략으로 바로 연결할 수 있습니다.</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("학생을 이해하는 4가지 렌즈")
    lens_df = pd.DataFrame(
        [
            {"구분": "내면의 마음 상태", "변수": "수학불안", "의미": "수학 상황에서 느끼는 긴장과 걱정"},
            {"구분": "내면의 마음 상태", "변수": "자기효능감", "의미": "수학을 해낼 수 있다는 자신감"},
            {"구분": "내면의 마음 상태", "변수": "수학흥미", "의미": "수학에 대한 흥미와 참여 의지"},
            {"구분": "실제적 참여", "변수": "학습태도", "의미": "계획, 성실성, 질문, 집중 등 학습 행동"},
        ]
    )
    st.dataframe(lens_df, use_container_width=True, hide_index=True)
    st.caption("정서·동기 변수는 학생의 마음 상태를, 학습태도는 실제 수업 참여를 보여줍니다.")

    st.subheader("변수 조합별 최적 K값")
    combo_df = pd.DataFrame(
        [
            {"조합": "수학불안 + 자기효능감 + 수학흥미 + 학습태도", "추천 K": 2, "해석": "가장 종합적이고 안정적인 구분"},
            {"조합": "수학불안 + 자기효능감", "추천 K": 8, "해석": "학생 차이가 매우 세분화되어 나타남"},
            {"조합": "수학불안 + 자기효능감 + 수학흥미", "추천 K": 7, "해석": "정서·동기 차이가 복잡하게 나타남"},
            {"조합": "수학불안 + 수학흥미", "추천 K": 6, "해석": "불안과 흥미 조합에 다양한 유형 존재"},
            {"조합": "수학불안 + 학습태도", "추천 K": 6, "해석": "불안과 실제 학습행동의 조합이 다양함"},
            {"조합": "수학불안 + 자기효능감 + 학습태도", "추천 K": 2, "해석": "학습 지원 관점에서 안정적 구분"},
            {"조합": "수학불안 + 수학흥미 + 학습태도", "추천 K": 2, "해석": "학습 참여 관점에서 안정적 구분"},
        ]
    )
    st.dataframe(combo_df, use_container_width=True, hide_index=True)
    st.caption("정서·동기 변수만 사용하면 K가 커지고, 학습태도를 포함하면 K=2로 안정되는 경향이 나타납니다.")

    fig = px.bar(
        combo_df,
        x="추천 K",
        y="조합",
        orientation="h",
        text="추천 K",
        title="변수 조합별 추천 K값",
        labels={"추천 K": "추천 K", "조합": "변수 조합"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=20, r=30, t=60, b=30),
        yaxis={"categoryorder": "array", "categoryarray": combo_df["조합"].tolist()[::-1]},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("K값이 클수록 학생 유형이 세분화됩니다. 하지만 수업 전략으로 쓰기에는 단순하고 안정적인 K=2가 더 적합합니다.")

    st.subheader("데이터를 관통하는 핵심 패턴")
    st.markdown(
        """
<div class="soft-card">
  <div class="card-title">마음 상태 + 학습태도 = K의 안정화</div>
  <div class="card-text">
    수학불안·자기효능감·흥미만 보면 학생 차이가 복잡하게 나뉩니다.<br>
    그러나 학습태도를 함께 고려하면 학생 집단은 <b>수업에 비교적 잘 참여하는 집단</b>과 <b>수학불안이 높고 지원이 필요한 집단</b>으로 안정적으로 정리됩니다.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("K=2의 수업적 의미")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
<div class="cluster-card">
  <div class="cluster-stripe" style="background:#0d9e75"></div>
  <div class="cluster-name">수업에 비교적 잘 참여하는 집단</div>
  <div class="cluster-sub">수학에 안정적으로 참여하며, 현재의 학습 패턴을 유지할 수 있는 그룹입니다.</div>
  <span class="badge badge-green">유지·확장 지원</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
<div class="cluster-card">
  <div class="cluster-stripe" style="background:#e03131"></div>
  <div class="cluster-name">수학불안이 높고 지원이 필요한 집단</div>
  <div class="cluster-sub">심리적 불안을 낮추고, 교사의 적극적 개입과 맞춤형 지지가 필요한 그룹입니다.</div>
  <span class="badge badge-red">우선 지원 필요</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="callout">
  <b>최종 제언</b><br>
  해석 가능성과 수업 적용 가능성을 고려하여, 4개 변수 전체를 포함한 <b>K=2 군집 모델</b>을 권장합니다.
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("원본 PDF 리포트 보기", expanded=False):
        display_pdf_file("Student_Clustering_via_Variable_Selection.pdf")


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 분석 파일 업로드")
    uploaded = st.file_uploader("엑셀 또는 CSV 파일", type=["xlsx", "xls", "csv"])
    st.markdown("---")
    st.markdown("### ⚙️ 분석 설정")
    selected_vars = []
    for var, info in FACTOR_INFO.items():
        if st.checkbox(info["name"], value=True, key=f"var_{var}"):
            selected_vars.append(var)
    k_mode = st.radio("K 결정 방식", ["실루엣 점수로 자동 추천", "직접 선택"], index=0)
    k_manual = st.slider("직접 선택 K", 2, 6, 2, disabled=(k_mode == "실루엣 점수로 자동 추천"))
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    st.markdown("---")
    st.caption("기본 흐름은 구쌤기말 노트북의 분석 절차를 따르며, 화면 구성은 보고서형 HTML의 카드형 레이아웃을 반영했습니다.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if uploaded is None:
    report_header()
    st.markdown(
        """
<div class="soft-card">
  <div class="small-label">How to start</div>
  <div class="card-title">왼쪽에서 설문 응답 파일을 업로드하면 분석이 시작됩니다.</div>
  <div class="card-text">
    파일에는 A문항(수학불안), E문항(자기효능감), I문항(수학흥미), T문항(학습태도)이 포함되어 있으면 됩니다.
    앱은 문항을 자동 인식하고 T6 문항은 역채점한 뒤 하위요인 평균을 계산합니다.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="factor-card"><div class="factor-icon" style="background:#fff0f0;color:#e03131">A</div><b>수학불안</b><br><span style="color:#4e5968;font-size:13px">긴장·걱정·회피</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="factor-card"><div class="factor-icon" style="background:#eef3fd;color:#1b64da">E</div><b>자기효능감</b><br><span style="color:#4e5968;font-size:13px">자신감·해낼 수 있음</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="factor-card"><div class="factor-icon" style="background:#e6f7f1;color:#0d9e75">I</div><b>수학흥미</b><br><span style="color:#4e5968;font-size:13px">흥미·성취감·참여</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="factor-card"><div class="factor-icon" style="background:#fff8e6;color:#c47d0e">T</div><b>학습태도</b><br><span style="color:#4e5968;font-size:13px">계획·성실성·집중</span></div>', unsafe_allow_html=True)
    st.stop()

try:
    df_loaded = read_uploaded_file(uploaded)
    k_value = None if k_mode == "실루엣 점수로 자동 추천" else k_manual
    res = run_analysis(df_loaded, selected_vars, k_value, int(random_state))
except Exception as e:
    report_header()
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()

report_header(n=len(res.df_clean), k=res.k)

# Summary cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("전체 응답 수", f"{len(res.df_raw):,}명")
m2.metric("분석 가능 응답", f"{len(res.df_clean):,}명")
m3.metric("분석 변수", f"{len(res.selected_vars)}개")
m4.metric("선택 K", f"K={res.k}", delta=f"추천 K={res.k_recommended}" if res.k_recommended else None)

if res.k_recommended and res.k != res.k_recommended:
    st.markdown(
        f"""
<div class="warning-callout">
  현재 선택 K는 <b>{res.k}</b>이고, 실루엣 점수 기준 추천 K는 <b>{res.k_recommended}</b>입니다.
  보고서에서는 두 값을 함께 비교하면 더 설득력 있게 설명할 수 있습니다.
</div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs([
    "① 연구 개요·데이터",
    "② 문항·전처리",
    "③ K 탐색",
    "④ K탐색 결과",
    "⑤ 군집 시각화",
    "⑥ 결론·지도 방안",
])

# ── Tab 1
with tabs[0]:
    section(
        "Research Overview",
        "분석 목적과 보고 흐름",
        "이 분석은 학생의 수학 성취를 단순 점수로만 보지 않고, 수학불안·자기효능감·흥미·학습태도라는 네 가지 학습 경험 요인을 함께 살펴봅니다. "
        "K-평균 군집화는 비슷한 응답 패턴을 보이는 학생들을 같은 집단으로 묶어, 집단별 맞춤형 수업 지원 방향을 도출하는 데 활용됩니다.",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="soft-card"><div class="small-label">Analysis Story</div><div class="card-title">이 웹앱으로 확인할 수 있는 것</div><div class="card-text">① 학생들이 어떤 학습자 유형으로 나뉘는지<br>② 유형별 수학불안·효능감·흥미·태도 차이가 무엇인지<br>③ 군집을 나누는 적절한 K 값은 무엇인지<br>④ 각 유형에 맞는 수업 지원 방향은 무엇인지</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card"><div class="small-label">Method</div><div class="card-title">분석 방법</div><div class="card-text">문항 자동 인식 → T6 역채점 → 하위요인 평균 산출 → 표준화 → Elbow/실루엣 검토 → K-평균 군집화 → PCA·레이더·히트맵 시각화 → 지도 방안 도출</div></div>', unsafe_allow_html=True)

    st.subheader("4개 하위요인")
    cols = st.columns(4)
    for col, (var, info) in zip(cols, FACTOR_INFO.items()):
        found = len(res.item_map[var])
        with col:
            st.markdown(
                f"""
<div class="factor-card">
  <div class="factor-icon" style="background:{info['bg']};color:{info['color']}">{info['short']}</div>
  <b>{info['name']}</b><br>
  <span style="color:#4e5968;font-size:13px">{info['desc']}</span><br>
  <span style="color:#8b95a1;font-size:12px">{found}문항 인식</span>
</div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("원자료 미리보기", expanded=False):
        st.dataframe(res.df_raw.head(30), use_container_width=True)

    with st.expander("하위요인 평균 데이터 미리보기", expanded=False):
        show_cols = [c for c in ["cluster"] + res.selected_vars if c in res.result_df.columns]
        st.dataframe(res.result_df[show_cols].head(30).rename(columns=VAR_LABELS), use_container_width=True)

# ── Tab 2
with tabs[1]:
    section(
        "Survey & Preprocessing",
        "문항 인식과 하위요인 평균 산출",
        "앱은 열 이름의 A, E, I, T 접두어를 기준으로 문항을 자동 인식합니다. 학습태도 문항 중 T6은 ‘수학을 포기하고 싶다’와 같은 부정 문항이므로 5점 척도 기준으로 역채점합니다.",
    )

    for var, cols in res.item_map.items():
        info = FACTOR_INFO[var]
        with st.expander(f"{info['short']} · {info['name']} 문항 {len(cols)}개", expanded=(var == "math_anxiety_mean")):
            if not cols:
                st.warning("해당 접두어 문항을 찾지 못했습니다.")
            else:
                rows = []
                for c in cols:
                    code = get_item_code(c)
                    rows.append({
                        "문항코드": code,
                        "문항": c,
                        "처리": "역채점" if code in REVERSE_ITEMS else "정방향",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("결측치 확인")
    missing = res.df_factor[res.selected_vars].isna().sum().rename(index=VAR_LABELS).reset_index()
    missing.columns = ["요인", "결측치 수"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown(
        """
<div class="callout">
  <b>해석 팁</b><br>
  표준화 점수는 전체 평균을 0으로 둔 상대적 위치입니다. 예를 들어 수학불안 Z-score가 +0.7이면 전체 평균보다 높은 불안을,
  자기효능감 Z-score가 -1.0이면 전체 평균보다 낮은 자신감을 의미합니다.
</div>
        """,
        unsafe_allow_html=True,
    )

# ── Tab 3
with tabs[2]:
    section(
        "Finding K",
        "군집 수 K를 정하는 과정",
        "노트북의 흐름처럼 Elbow 방법과 실루엣 점수를 함께 확인합니다. Elbow는 Inertia가 급격히 줄어들다가 완만해지는 지점을 보고, 실루엣 점수는 군집 내부 응집도와 군집 간 분리도를 함께 판단합니다.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_elbow(res), use_container_width=True)
    with c2:
        st.plotly_chart(fig_silhouette(res), use_container_width=True)

    if not res.silhouette_df.empty:
        st.subheader("실루엣 점수 표")
        sil_table = res.silhouette_df.copy()
        sil_table["Silhouette"] = sil_table["Silhouette"].round(3)
        st.dataframe(
            sil_table.rename(columns={"K": "군집 수 K", "Silhouette": "실루엣 점수"}),
            use_container_width=True,
            hide_index=True,
        )

    if res.k_recommended:
        best_row = res.silhouette_df.loc[res.silhouette_df["Silhouette"].idxmax()]
        best_k = int(best_row["K"])
        best_score = float(best_row["Silhouette"])
        st.markdown(
            f"""
<div class="callout">
  <b>보고서 문장 예시</b><br>
  실루엣 점수 표를 보면, <b>K={best_k}</b>일 때 실루엣 점수가 <b>{best_score:.3f}</b>로 가장 높게 나타났습니다.
  이는 군집의 응집도와 분리도가 가장 이상적인 K 값으로 <b>K={best_k}</b>를 추천한다는 의미입니다.
  즉, <b>{best_k}</b>개의 군집으로 분류하는 것이 데이터의 특성을 가장 잘 반영한다고 해석할 수 있습니다.
</div>
            """,
            unsafe_allow_html=True,
        )

# ── Tab 4
with tabs[3]:
    render_k_exploration_result_tab()

# ── Tab 5
with tabs[4]:
    section(
        "Cluster Visualization",
        "군집 프로파일을 여러 시각 요소로 확인",
        "군집 인원, 원점수 평균, 표준화 평균, 레이더 차트, PCA 산점도, 중심 이동 과정, 수학불안-학습태도 관계를 한 화면 흐름으로 확인합니다.",
    )

    st.plotly_chart(fig_cluster_counts(res), use_container_width=True)
    st.caption("군집별 학생 수를 비교합니다. 특정 군집의 인원이 많다면 그 유형에 맞는 수업 지원을 우선적으로 설계할 필요가 있습니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_heatmap(res.profile_raw, "원점수 평균 히트맵", zscore=False), use_container_width=True)
        st.caption("각 군집의 실제 5점 척도 평균을 보여줍니다. 학생들이 설문에서 실제로 어느 정도 점수를 보였는지 직관적으로 확인할 수 있습니다.")
    with c2:
        st.plotly_chart(fig_heatmap(res.profile_scaled, "표준화 평균 히트맵", zscore=True), use_container_width=True)
        st.caption("전체 평균을 0으로 두고 군집별 상대적 차이를 보여줍니다. +값은 평균보다 높고, -값은 평균보다 낮다는 뜻입니다.")

    st.plotly_chart(fig_radar(res), use_container_width=True)
    st.caption("군집별 표준화 평균을 한눈에 비교하는 차트입니다. 선이 바깥쪽으로 갈수록 해당 요인이 평균보다 높고, 안쪽으로 갈수록 평균보다 낮습니다.")

    st.plotly_chart(fig_pca(res), use_container_width=True)
    st.caption("여러 설문 요인을 2차원 평면으로 줄여 학생들의 분포를 보여줍니다. 가까이 모인 점들은 응답 패턴이 비슷하고, 멀리 떨어진 점들은 응답 특성이 다르다고 볼 수 있습니다.")

    reg_fig = fig_anxiety_attitude(res)
    if reg_fig:
        st.plotly_chart(reg_fig, use_container_width=True)
        st.caption("수학불안이 높아질수록 학습태도가 어떻게 달라지는지 살펴보는 산점도입니다. 추세선이 아래로 향하면 불안이 높을수록 학습태도가 낮아지는 경향이 있음을 의미합니다.")

    st.subheader("K-평균 군집화 과정: Centroid 변화")
    st.caption("PCA 2차원 공간에서 군집 중심이 이동하고 학생 점들이 다시 배정되는 과정을 단계별로 보여줍니다.")
    step = st.slider("Centroid 변화 단계", 0, 5, 2)
    st.plotly_chart(fig_centroid_step(res, step), use_container_width=True)
    st.caption("K-평균 군집화는 임의의 중심점에서 출발해, 학생들을 가까운 중심에 배정하고 중심을 다시 계산하는 과정을 반복합니다. 중심점이 거의 움직이지 않으면 최종 군집이 결정됩니다.")

# ── Tab 6
with tabs[5]:
    section(
        "Conclusion & Teaching Strategies",
        "군집별 특징과 맞춤형 지도 방안",
        "군집명은 학생을 고정적으로 낙인찍기 위한 이름이 아니라, 현재 수학 학습 경험을 이해하고 지원 방향을 찾기 위한 임시적·탐색적 이름입니다.",
    )

    cluster_cols = st.columns(min(res.k, 3))
    # If more than 3 clusters, render in rows
    clusters = list(res.profile_scaled.index)
    for idx, cluster in enumerate(clusters):
        interp = interpret_cluster(res.profile_scaled.loc[cluster], res.profile_raw.loc[cluster], res.selected_vars)
        target_col = cluster_cols[idx % len(cluster_cols)]
        with target_col:
            badges = ""
            for var in res.selected_vars:
                z = res.profile_scaled.loc[cluster, var]
                cls = "badge-green" if (z >= 0 and var != "math_anxiety_mean") or (z < 0 and var == "math_anxiety_mean") else "badge-red"
                badges += f'<span class="badge {cls}">{VAR_LABELS[var]} {z:+.2f}</span>'
            st.markdown(
                f"""
<div class="cluster-card">
  <div class="cluster-stripe" style="background:{interp['color']}"></div>
  <div style="font-size:12px;font-weight:700;color:{interp['color']};margin-top:6px;">군집 {cluster}</div>
  <div class="cluster-name">{interp['title']}</div>
  <div class="cluster-sub">{interp['subtitle']}</div>
  {badges}
  <hr>
  <div style="font-size:13px;color:#4e5968;line-height:1.7">
    <b>강점</b>: {interp['strengths']}<br>
    <b>지원 필요</b>: {interp['needs']}<br><br>
    <b>지도 방안</b><br>{interp['strategy']}
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("군집별 평균표")
    st.dataframe(
        res.profile_scaled.rename(columns=VAR_LABELS).style.format("{:+.2f}"),
        use_container_width=True,
    )

    s1_fig = fig_s1_keywords(res)
    if s1_fig:
        st.plotly_chart(s1_fig, use_container_width=True)
        s1_col = get_s1_col(res.df_raw)
        with st.expander("S1 자유응답 원문 보기", expanded=False):
            show = res.result_df[["cluster"]].copy()
            show["S1 자유응답"] = res.df_raw.loc[res.result_df.index, s1_col].astype(str)
            show = show[show["S1 자유응답"].str.strip().ne("") & show["S1 자유응답"].ne("nan")]
            st.dataframe(show.sort_values("cluster"), use_container_width=True)

    st.markdown(
        """
<div class="callout">
  <b>종합 시사점</b><br>
  같은 성취 수준의 학생이라도 수학불안, 자기효능감, 흥미, 학습태도 조합은 다를 수 있습니다.
  따라서 본 결과는 점수 중심의 획일적 보충보다, 정서·동기·행동 특성을 고려한 수업 설계에 활용할 때 의미가 큽니다.
</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("다운로드")
    result_download = res.result_df.copy()
    csv = result_download.to_csv(index=False).encode("utf-8-sig")
    st.download_button("군집 결과 CSV 다운로드", csv, file_name="math_cluster_results.csv", mime="text/csv")

    md = report_markdown(res)
    st.download_button(
        "보고서 요약 Markdown 다운로드",
        md.encode("utf-8-sig"),
        file_name="math_cluster_report_summary.md",
        mime="text/markdown",
    )
