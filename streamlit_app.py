# app.py
# Streamlit web app: K-평균 군집화를 활용한 수학 학습자 유형 분석
# 실행: streamlit run app.py

import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="수학 학습자 유형 분석",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# 1. 기본 설정
# -----------------------------
FACTOR_ITEMS: Dict[str, List[str]] = {
    "math_anxiety_mean": [
        "A1. 수학 시간에 어려운 문제가 나오면 긴장된다.",
        "A2. 수학 시험을 생각하면 걱정이 앞선다.",
        "A3. 수학 문제를 풀다가 막히면 당황해서 더 생각하기 어렵다.",
        "A4. 친구들 앞에서 수학 문제를 풀거나 발표하는 것이 부담스럽다.",
        "A5. 수학 성적이 잘 나오지 않을까 봐 불안하다.",
        "A6. 수학 시간에 선생님이 질문할까 봐 긴장될 때가 있다.",
        "A7. 수학 문제를 보면 시작하기 전부터 어렵게 느껴진다.",
        "A8. 수학 시험 중 시간이 부족할 것 같으면 머리가 하얘진다.",
    ],
    "math_self_efficacy_mean": [
        "E1. 나는 노력하면 수학 실력을 향상시킬 수 있다고 생각한다.",
        "E2. 처음에는 어려운 수학 문제도 차근차근 생각하면 해결할 수 있다.",
        "E3. 수학 문제를 틀려도 다시 도전할 수 있다.",
        "E4. 나는 수학 수업 내용을 이해할 수 있다는 자신감이 있다.",
        "E5. 새로운 수학 개념도 설명을 들으면 이해할 수 있다고 생각한다.",
        "E6. 수학 과제를 끝까지 해낼 수 있다.",
        "E7. 시험에서 모르는 문제가 나와도 아는 내용을 활용해 보려고 한다.",
        "E8. 나는 수학 공부 방법을 스스로 조절할 수 있다.",
    ],
    "math_interest_mean": [
        "I1. 수학 문제를 해결했을 때 성취감을 느낀다.",
        "I2. 수학 시간에 배우는 내용이 흥미롭다고 느낄 때가 있다.",
        "I3. 새로운 수학 개념을 배우는 것이 재미있다.",
        "I4. 수학이 실생활이나 다른 분야와 연결된다는 점이 흥미롭다.",
        "I5. 어려운 문제를 고민해 보는 과정이 의미 있다고 생각한다.",
        "I6. 수학 관련 활동이나 탐구 과제에 참여해 보고 싶다.",
        "I7. 수학을 잘하면 앞으로 도움이 될 것이라고 생각한다.",
        "I8. 수학 시간에 적극적으로 참여하고 싶다.",
    ],
    "learning_attitude_mean": [
        "T1. 나는 수학 공부 계획을 세우고 실천하는 편이다.",
        "T2. 수학 숙제나 과제를 성실히 하는 편이다.",
        "T3. 모르는 수학 문제가 있으면 질문하거나 찾아보는 편이다.",
        "T4. 수학 공부를 할 때 집중이 잘 되는 편이다.",
        "T5. 수학 수업에서 친구들과 함께 문제를 해결하는 것이 도움이 된다.",
        "T6. 수학을 포기하고 싶다고 느낄 때가 있다.",
    ],
}

FACTOR_LABELS = {
    "math_anxiety_mean": "수학불안",
    "math_self_efficacy_mean": "자기효능감",
    "math_interest_mean": "수학흥미",
    "learning_attitude_mean": "학습태도",
}

NEGATIVE_ITEMS = ["T6. 수학을 포기하고 싶다고 느낄 때가 있다."]

GUIDE_TEXT = {
    "취약형": {
        "조건": "수학불안이 높고 자기효능감·흥미·학습태도가 낮은 집단",
        "지도방안": [
            "쉬운 성공 경험을 자주 제공하고, 단계형 문제로 시작합니다.",
            "정답 여부보다 시도 과정과 접근 방법을 칭찬합니다.",
            "공개 발표보다 짝 활동·소그룹 활동을 먼저 활용합니다.",
            "평가 부담을 낮추고 오답을 학습 자료로 다루는 분위기를 만듭니다.",
        ],
    },
    "자신감·흥미형": {
        "조건": "수학불안이 낮고 자기효능감·흥미·학습태도가 높은 집단",
        "지도방안": [
            "심화 문제, 프로젝트형 과제, 전공 연계 탐구 과제를 제공합니다.",
            "또래 멘토 역할을 부여하여 학급 내 긍정적 학습 문화를 확산합니다.",
            "자기주도 학습과 발표 기회를 확대합니다.",
        ],
    },
    "무관심형": {
        "조건": "불안은 높지 않지만 흥미와 학습태도가 낮은 집단",
        "지도방안": [
            "수학을 전공·자격증·직업 상황과 연결해 필요성을 느끼게 합니다.",
            "계산 반복보다 실생활 문제, 데이터 해석, 직무 상황 문제를 활용합니다.",
            "짧은 활동 중심 수업으로 참여 진입 장벽을 낮춥니다.",
        ],
    },
    "잠재성장형": {
        "조건": "흥미는 있으나 불안이 높거나 자기효능감이 충분하지 않은 집단",
        "지도방안": [
            "도전 과제를 주되, 중간 힌트와 피드백을 충분히 제공합니다.",
            "작은 발표, 선택형 과제 등 부담이 낮은 표현 기회를 제공합니다.",
            "성취 경험을 기록하게 하여 자신감으로 연결합니다.",
        ],
    },
    "평균형": {
        "조건": "세부 요인이 전반적으로 평균 부근인 집단",
        "지도방안": [
            "수업 참여 기회를 꾸준히 제공하고, 개별 피드백을 강화합니다.",
            "기본 개념 확인과 전공 연계 활동을 균형 있게 운영합니다.",
            "어느 요인이 약한지 추가로 확인해 보완합니다.",
        ],
    },
}


# -----------------------------
# 2. 함수 정의
# -----------------------------
@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("CSV, XLSX, XLS 파일만 업로드할 수 있습니다.")


def normalize_colname(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip()


def find_matching_column(df_columns: List[str], item: str) -> str | None:
    """정확히 일치하지 않아도 A1., E2. 같은 문항 코드가 맞으면 찾는다."""
    if item in df_columns:
        return item

    code_match = re.match(r"^([AEIT]\d+)\.", item)
    if code_match:
        code = code_match.group(1)
        pattern = re.compile(rf"^{re.escape(code)}[\.\s\)]")
        for col in df_columns:
            if pattern.search(str(col).strip()):
                return col

    item_norm = normalize_colname(item)
    for col in df_columns:
        if normalize_colname(col) == item_norm:
            return col
    return None


def compute_factor_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    df = df.copy()
    col_list = list(df.columns)
    used_columns: Dict[str, List[str]] = {}
    missing_items: List[str] = []

    result = pd.DataFrame(index=df.index)

    # 식별용 컬럼 후보 자동 탐색
    id_candidates = [
        "번호 또는 익명 코드",
        "익명 코드",
        "번호",
        "학번",
        "이름",
        "index",
    ]
    id_col = None
    for cand in id_candidates:
        for col in col_list:
            if cand in str(col):
                id_col = col
                break
        if id_col is not None:
            break

    if id_col is not None:
        result["student_id"] = df[id_col].astype(str)
    else:
        result["student_id"] = df.index.astype(str)

    for factor, items in FACTOR_ITEMS.items():
        matched_cols = []
        for item in items:
            matched = find_matching_column(col_list, item)
            if matched is not None:
                matched_cols.append(matched)
            else:
                missing_items.append(item)

        used_columns[factor] = matched_cols

        if matched_cols:
            temp = df[matched_cols].copy()
            for col in matched_cols:
                temp[col] = pd.to_numeric(temp[col], errors="coerce")

            # T6 부정문항 역채점: 5점 척도 기준 6 - 점수
            if factor == "learning_attitude_mean":
                for neg_item in NEGATIVE_ITEMS:
                    neg_col = find_matching_column(matched_cols, neg_item)
                    if neg_col in temp.columns:
                        temp[neg_col] = 6 - temp[neg_col]

            result[factor] = temp.mean(axis=1)
        else:
            result[factor] = np.nan

    return result, used_columns, missing_items


def run_kmeans(df_factor: pd.DataFrame, selected_vars: List[str], k: int):
    X = df_factor[selected_vars].copy()
    for col in selected_vars:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    valid_idx = X.dropna().index
    X_valid = X.loc[valid_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = df_factor.loc[valid_idx].copy()
    result["cluster"] = labels + 1

    scaled_df = pd.DataFrame(X_scaled, columns=selected_vars, index=valid_idx)
    scaled_df["cluster"] = labels + 1

    profile_raw = result.groupby("cluster")[selected_vars].mean()
    profile_scaled = scaled_df.groupby("cluster")[selected_vars].mean()
    counts = result["cluster"].value_counts().sort_index()

    return result, scaled_df, profile_raw, profile_scaled, counts, model, scaler


def get_elbow_silhouette(df_factor: pd.DataFrame, selected_vars: List[str], k_min: int, k_max: int) -> pd.DataFrame:
    X = df_factor[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = []
    for k in range(k_min, k_max + 1):
        if k >= len(X_scaled):
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        sil = silhouette_score(X_scaled, labels) if k > 1 and len(set(labels)) > 1 else np.nan
        rows.append({"K": k, "Inertia": inertia, "Silhouette": sil})
    return pd.DataFrame(rows)


def recommend_k_by_silhouette(eval_df: pd.DataFrame) -> int | None:
    valid = eval_df.dropna(subset=["Silhouette"])
    if valid.empty:
        return None
    return int(valid.loc[valid["Silhouette"].idxmax(), "K"])


def make_silhouette_report_sentence(eval_df: pd.DataFrame) -> str:
    recommended_k = recommend_k_by_silhouette(eval_df)
    if recommended_k is None:
        return "실루엣 점수를 계산할 수 없어 Elbow 그래프와 교육적 해석 가능성을 함께 고려하여 K 값을 결정하였다."

    best_score = float(eval_df.loc[eval_df["K"] == recommended_k, "Silhouette"].iloc[0])
    return (
        f"실루엣 점수 표를 보면, K={recommended_k}일 때 실루엣 점수가 {best_score:.3f}으로 가장 높게 나타났다. "
        f"이는 군집의 응집도와 분리도를 함께 고려했을 때 K={recommended_k}가 가장 적절한 군집 수임을 의미한다. "
        f"따라서 본 분석에서는 {recommended_k}개의 군집으로 분류하는 것이 데이터의 특성을 가장 잘 반영한다고 해석할 수 있다."
    )


def kmeans_centroid_history(X_scaled: np.ndarray, k: int, max_iter: int = 10) -> List[np.ndarray]:
    """K-평균 군집화의 중심점 이동 과정을 시각화하기 위한 간단한 구현."""
    centers, _ = kmeans_plusplus(X_scaled, n_clusters=k, random_state=42)
    history = [centers.copy()]

    for _ in range(max_iter):
        distances = np.linalg.norm(X_scaled[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_idx in range(k):
            members = X_scaled[labels == cluster_idx]
            if len(members) > 0:
                new_centers[cluster_idx] = members.mean(axis=0)
        history.append(new_centers.copy())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return history


def classify_cluster(row: pd.Series) -> str:
    anx = row.get("math_anxiety_mean", np.nan)
    eff = row.get("math_self_efficacy_mean", np.nan)
    inter = row.get("math_interest_mean", np.nan)
    attitude = row.get("learning_attitude_mean", np.nan)

    positive_vals = [v for v in [eff, inter, attitude] if pd.notna(v)]
    positive_mean = np.mean(positive_vals) if positive_vals else np.nan

    if pd.notna(anx) and pd.notna(positive_mean):
        if anx >= 0.6 and positive_mean <= -0.4:
            return "취약형"
        if anx <= -0.4 and positive_mean >= 0.5:
            return "자신감·흥미형"
        if anx >= 0.4 and inter >= 0.2:
            return "잠재성장형"
        if anx <= 0.3 and positive_mean <= -0.4:
            return "무관심형"
    return "평균형"


def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FACTOR_LABELS)


# -----------------------------
# 3. 화면 구성
# -----------------------------
st.title("📊 K-평균 군집화를 활용한 수학 학습자 유형 분석")
st.caption("수학불안 · 자기효능감 · 흥미 · 학습태도 설문 데이터를 업로드하면 학생 집단을 자동으로 유형화하고 시각화합니다.")

with st.expander("이 웹앱으로 확인할 수 있는 것", expanded=True):
    st.markdown(
        """
        - 우리 학교 학생들이 어떤 수학 학습자 유형인지 파악
        - 수학을 어려워하는 이유가 불안, 자신감 부족, 흥미 부족 중 무엇과 관련되는지 확인
        - 군집별 맞춤형 수업 전략 마련
        - 특성화고 학생에게 맞는 전공·진로 연계 수학 수업 방향 탐색
        """
    )

uploaded_file = st.sidebar.file_uploader(
    "설문 응답 파일 업로드",
    type=["csv", "xlsx", "xls"],
    help="구글폼 응답을 스프레드시트에서 CSV 또는 엑셀로 내려받아 업로드하세요.",
)

st.sidebar.header("분석 설정")
default_vars = list(FACTOR_ITEMS.keys())
selected_vars = st.sidebar.multiselect(
    "군집화에 사용할 변수",
    options=default_vars,
    default=["math_anxiety_mean", "math_self_efficacy_mean", "math_interest_mean"],
    format_func=lambda x: FACTOR_LABELS.get(x, x),
)
k = st.sidebar.slider("군집 수 K", min_value=2, max_value=8, value=3, step=1)
show_attitude = "learning_attitude_mean" in selected_vars

if uploaded_file is None:
    st.info("왼쪽 사이드바에서 구글폼 응답 CSV 또는 엑셀 파일을 업로드해 주세요.")
    st.markdown(
        """
        ### 파일 준비 방법
        1. Google Forms → 응답 → 스프레드시트로 연결  
        2. Google Sheets → 파일 → 다운로드 → `.xlsx` 또는 `.csv`  
        3. 이 웹앱 왼쪽에서 파일 업로드  
        """
    )
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

factor_df, used_columns, missing_items = compute_factor_scores(raw_df)

st.subheader("1. 데이터 확인")
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 응답 수", f"{len(raw_df):,}명")
c2.metric("분석 가능 변수 수", f"{len([v for v in default_vars if factor_df[v].notna().any()])}개")
c3.metric("선택 변수 수", f"{len(selected_vars)}개")
c4.metric("선택 K", f"{k}개")

with st.expander("원자료 미리보기"):
    st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("문항 인식 결과 확인"):
    for factor, cols in used_columns.items():
        st.write(f"**{FACTOR_LABELS[factor]}**: {len(cols)}개 문항 인식")
        st.caption(", ".join(map(str, cols)) if cols else "인식된 문항 없음")
    if missing_items:
        st.warning("일부 문항은 파일에서 찾지 못했습니다. 문항 제목이 다르면 A1, E1, I1, T1 같은 코드가 제목 앞에 들어가도록 수정하면 인식률이 높아집니다.")

st.subheader("2. 하위요인 평균 점수")
display_factor_df = factor_df.copy()
st.dataframe(rename_for_display(display_factor_df).head(30), use_container_width=True)

if len(selected_vars) < 2:
    st.error("K-평균 군집화를 실행하려면 최소 2개 이상의 변수를 선택해야 합니다.")
    st.stop()

missing_selected = [v for v in selected_vars if factor_df[v].isna().all()]
if missing_selected:
    st.error("선택한 변수 중 계산할 수 없는 변수가 있습니다: " + ", ".join([FACTOR_LABELS[v] for v in missing_selected]))
    st.stop()

valid_n = factor_df[selected_vars].dropna().shape[0]
if valid_n <= k:
    st.error(f"결측치 제거 후 분석 가능 인원이 {valid_n}명입니다. K 값을 더 작게 설정하세요.")
    st.stop()

# -----------------------------
# 4. Elbow / Silhouette
# -----------------------------
st.subheader("3. K 값 검토: Elbow & Silhouette")
max_k_allowed = min(8, valid_n - 1)
eval_df = get_elbow_silhouette(factor_df, selected_vars, 2, max_k_allowed)

col1, col2 = st.columns(2)
with col1:
    fig_elbow = px.line(eval_df, x="K", y="Inertia", markers=True, title="Elbow 그래프")
    fig_elbow.update_layout(xaxis=dict(dtick=1), height=380)
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = px.line(eval_df, x="K", y="Silhouette", markers=True, title="Silhouette 점수")
    fig_sil.update_layout(xaxis=dict(dtick=1), height=380)
    st.plotly_chart(fig_sil, use_container_width=True)

st.caption("Elbow는 꺾이는 지점을, Silhouette은 상대적으로 높은 값을 참고합니다. 교육적 해석 가능성도 함께 고려해 K를 결정하세요.")

with st.expander("K 탐색 결과표 및 보고서 문장 예시", expanded=True):
    eval_display = eval_df.copy()
    eval_display["Inertia"] = eval_display["Inertia"].round(3)
    eval_display["Silhouette"] = eval_display["Silhouette"].round(3)
    st.dataframe(eval_display, use_container_width=True, hide_index=True)
    st.markdown("**보고서 문장 예시**")
    st.info(make_silhouette_report_sentence(eval_df))

# -----------------------------
# 5. K-means 실행
# -----------------------------
clustered_df, scaled_df, profile_raw, profile_scaled, counts, model, scaler = run_kmeans(factor_df, selected_vars, k)

# 군집명 자동 추정
cluster_type_map = {}
for cluster_id, row in profile_scaled.iterrows():
    cluster_type_map[cluster_id] = classify_cluster(row)

clustered_df["cluster_type"] = clustered_df["cluster"].map(cluster_type_map)

st.subheader("4. 군집화 결과 요약")
m1, m2, m3 = st.columns(3)
m1.metric("분석에 사용된 학생 수", f"{len(clustered_df):,}명")
m2.metric("제외된 응답 수", f"{len(raw_df) - len(clustered_df):,}명")
m3.metric("군집 수", f"{k}개")

count_df = counts.reset_index()
count_df.columns = ["군집", "학생 수"]
count_df["비율(%)"] = (count_df["학생 수"] / count_df["학생 수"].sum() * 100).round(1)
count_df["자동 해석"] = count_df["군집"].map(cluster_type_map)

col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(count_df, use_container_width=True, hide_index=True)
with col2:
    fig_count = px.pie(
        count_df,
        names="군집",
        values="학생 수",
        title="군집별 학생 비율",
        hole=0.35,
    )
    st.plotly_chart(fig_count, use_container_width=True)

# -----------------------------
# 6. 군집 프로파일 시각화
# -----------------------------
st.subheader("5. 군집별 특성 시각화")

vis_tabs = st.tabs([
    "원점수 평균 히트맵",
    "표준화 평균 히트맵",
    "표준화 프로파일 레이더",
    "PCA 군집 2차원 분포",
    "수학불안-학습태도 관계",
    "Centroid 변화과정",
])

with vis_tabs[0]:
    raw_profile_display = rename_for_display(profile_raw.copy()).round(2)
    fig_raw_heat = px.imshow(
        raw_profile_display,
        text_auto=".2f",
        aspect="auto",
        title="군집별 원점수 평균 히트맵",
        labels=dict(x="요인", y="군집", color="5점 척도 평균"),
    )
    fig_raw_heat.update_layout(height=450)
    st.plotly_chart(fig_raw_heat, use_container_width=True)
    st.caption(
        "그림 설명: 각 군집의 실제 5점 척도 평균을 보여줍니다. 값이 높을수록 해당 군집 학생들이 그 요인을 더 강하게 경험한다는 뜻입니다. "
        "예를 들어 수학불안 평균이 높으면 수학 수업이나 평가 상황에서 긴장과 부담을 크게 느끼는 집단으로 해석할 수 있습니다."
    )
    st.dataframe(raw_profile_display, use_container_width=True)

with vis_tabs[1]:
    heat_df = rename_for_display(profile_scaled.copy())
    fig_heat = px.imshow(
        heat_df,
        text_auto=".2f",
        aspect="auto",
        title="군집별 표준화 평균 히트맵",
        labels=dict(x="요인", y="군집", color="표준화 점수"),
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "그림 설명: 표준화 점수는 전체 평균을 0으로 두고 군집별 특성을 비교한 값입니다. 0보다 크면 전체 학생 평균보다 높은 편, "
        "0보다 작으면 낮은 편입니다. 군집 간 상대적 차이를 비교할 때 원점수보다 더 유용합니다."
    )

with vis_tabs[2]:
    radar_df = profile_scaled.copy()
    categories = [FACTOR_LABELS.get(v, v) for v in selected_vars]
    fig_radar = go.Figure()
    for cluster_id, row in radar_df.iterrows():
        values = row[selected_vars].tolist()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"군집 {cluster_id}: {cluster_type_map.get(cluster_id, '')}",
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="표준화 군집 프로파일 레이더 차트",
        height=560,
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "그림 설명: 각 군집의 수학불안, 자기효능감, 흥미 등 요인별 강약을 한눈에 비교할 수 있습니다. "
        "바깥쪽으로 뻗은 요인은 해당 군집에서 상대적으로 높은 특성이고, 안쪽에 가까운 요인은 상대적으로 낮은 특성입니다."
    )

with vis_tabs[3]:
    X_scaled = scaled_df[selected_vars]
    pca = PCA(n_components=2, random_state=42)
    pca_xy = pca.fit_transform(X_scaled)
    pca_df = clustered_df[["student_id", "cluster", "cluster_type"]].copy()
    pca_df["PC1"] = pca_xy[:, 0]
    pca_df["PC2"] = pca_xy[:, 1]

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        symbol="cluster_type",
        hover_data=["student_id", "cluster_type"],
        title=f"PCA 군집 2차원 분포  |  설명분산: PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%",
    )
    fig_pca.update_layout(height=520)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(
        "그림 설명: 여러 설문 요인을 2개의 축으로 압축해 학생들의 군집 분포를 나타낸 그림입니다. "
        "점들이 서로 떨어져 있을수록 군집 간 특성 차이가 비교적 뚜렷하다고 볼 수 있고, 겹치는 부분이 많으면 군집 간 경계가 완만하다고 해석합니다."
    )

with vis_tabs[4]:
    if {"math_anxiety_mean", "learning_attitude_mean"}.issubset(clustered_df.columns) and clustered_df[["math_anxiety_mean", "learning_attitude_mean"]].notna().all(axis=1).sum() > 1:
        relation_df = clustered_df.copy()
        relation_df = relation_df.rename(columns=FACTOR_LABELS)
        fig_relation = px.scatter(
            relation_df,
            x="수학불안",
            y="학습태도",
            color="cluster",
            symbol="cluster_type",
            hover_data=["student_id", "cluster_type"],
            trendline="ols",
            title="수학불안과 학습태도의 관계",
        )
        fig_relation.update_layout(height=520, xaxis_title="수학불안 평균", yaxis_title="학습태도 평균")
        st.plotly_chart(fig_relation, use_container_width=True)
        st.caption(
            "그림 설명: 수학불안이 높을수록 학습태도가 어떻게 달라지는지 확인하는 산점도입니다. "
            "오른쪽 아래에 점이 많으면 불안이 높고 학습태도가 낮은 학생이 많다는 뜻이므로, 불안 완화와 작은 성공 경험 제공이 중요합니다. "
            "반대로 불안이 높아도 학습태도가 유지되는 학생은 잠재성장형으로 보고 적절한 피드백과 도전 기회를 줄 수 있습니다."
        )
    else:
        st.info("수학불안과 학습태도 평균이 모두 계산되어야 관계 그래프를 표시할 수 있습니다.")

with vis_tabs[5]:
    X_scaled_np = scaled_df[selected_vars].to_numpy()
    pca_for_centroid = PCA(n_components=2, random_state=42)
    pca_for_centroid.fit(X_scaled_np)
    history = kmeans_centroid_history(X_scaled_np, k=k, max_iter=10)

    rows = []
    for step, centers in enumerate(history):
        centers_2d = pca_for_centroid.transform(centers)
        for idx, xy in enumerate(centers_2d, start=1):
            rows.append({"반복 단계": step, "중심점": f"Centroid {idx}", "PC1": xy[0], "PC2": xy[1]})
    centroid_df = pd.DataFrame(rows)

    fig_centroid = px.line(
        centroid_df,
        x="PC1",
        y="PC2",
        color="중심점",
        text="반복 단계",
        markers=True,
        title="K-평균 군집화 과정: Centroid 변화과정",
    )
    fig_centroid.update_traces(textposition="top center")
    fig_centroid.update_layout(height=560, xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_centroid, use_container_width=True)
    st.caption(
        "그림 설명: K-평균 군집화에서 중심점이 반복 계산을 통해 이동하는 과정을 보여줍니다. "
        "초기 중심점에서 출발해 학생들이 가까운 중심점에 배정되고, 다시 군집 평균으로 중심점이 이동하는 과정을 반복합니다. "
        "중심점 이동이 거의 멈추면 군집 구성이 안정되었다고 볼 수 있습니다."
    )

# -----------------------------
# 7. 해석 및 지도방안
# -----------------------------
st.subheader("6. 군집별 해석 및 맞춤형 지도 방안")

for cluster_id in sorted(cluster_type_map):
    ctype = cluster_type_map[cluster_id]
    guide = GUIDE_TEXT.get(ctype, GUIDE_TEXT["평균형"])
    with st.expander(f"군집 {cluster_id} · {ctype}", expanded=True):
        st.write(f"**특징:** {guide['조건']}")
        profile_sentence = ", ".join(
            [f"{FACTOR_LABELS.get(v, v)} {profile_scaled.loc[cluster_id, v]:.2f}" for v in selected_vars]
        )
        st.caption(f"표준화 평균: {profile_sentence}")
        st.write("**수업 지원 방향**")
        for item in guide["지도방안"]:
            st.markdown(f"- {item}")

# -----------------------------
# 8. 학생별 결과 및 다운로드
# -----------------------------
st.subheader("7. 학생별 군집 결과")
result_display = clustered_df.copy()
result_display = result_display.rename(columns={
    "student_id": "학생ID",
    "cluster": "군집",
    "cluster_type": "군집유형",
    **FACTOR_LABELS,
})
st.dataframe(result_display, use_container_width=True)

download_df = result_display.copy()
st.download_button(
    label="📥 군집 결과 CSV 다운로드",
    data=make_download_csv(download_df),
    file_name="math_kmeans_cluster_result.csv",
    mime="text/csv",
)

st.divider()
st.markdown(
    """
    ### 보고서용 한 줄 해석
    본 분석은 학생들의 수학불안, 자기효능감, 흥미를 기준으로 수학 학습자 유형을 구분하고,
    각 유형에 적합한 맞춤형 수업 지원 방안을 마련하기 위한 것이다.
    """
)
