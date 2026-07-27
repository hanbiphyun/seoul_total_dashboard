import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="현황 분석 대시보드", layout="wide", page_icon="📊")

st.title("📊 서울시 자치구/법정동별 월세 및 건물 현황")
st.caption("선택한 지역의 월세 기준 평당 가격 중앙값, 보증금, 건물 나이, 지하 여부 및 가격 범위를 분석합니다.")

@st.cache_data
def load_data():
    file_id = "11apg9KTfsAtATCnQlSSpuXlTCPoAiNPM"  
    
    local_path = "data_final_light.parquet"
    df = pd.read_parquet(local_path)
    local_path = "data_final.csv"
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    if '전월세구분' in df.columns:
        df = df[df['전월세구분'] == '월세'].drop(columns=['전월세구분'])
    
    if '지하여부' in df.columns:
        df['지하여부_표시'] = df['지하여부'].apply(
            lambda x: '지하/반지하' if x in [1, True, '지하', '반지하', '지하/반지하'] else '지상'
        ).astype('category')
        df = df.drop(columns=['지하여부'])
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다. 오류 내용: {e}")
    st.stop()

st.sidebar.header("📍 지역 선택")
gu_list = ["전체"] + sorted(df['자치구명'].dropna().unique().tolist())
selected_gu = st.sidebar.selectbox("자치구 선택", gu_list)

if selected_gu == "전체":
    dong_list = ["전체"] + sorted(df['법정동명'].dropna().unique().tolist())
else:
    dong_list = ["전체"] + sorted(df[df['자치구명'] == selected_gu]['법정동명'].dropna().unique().tolist())

selected_dong = st.sidebar.selectbox("법정동 선택", dong_list)

sub_df = df.copy()
if selected_gu != "전체":
    sub_df = sub_df[sub_df['자치구명'] == selected_gu]
if selected_dong != "전체":
    sub_df = sub_df[sub_df['법정동명'] == selected_dong]

region_title = f"{selected_gu if selected_gu != '전체' else '서울시 전체'} {selected_dong if selected_dong != '전체' else ''}".strip()
st.subheader(f"📌 {region_title} 월세 분석 결과 ({len(sub_df):,} 건)")

if sub_df.empty:
    st.warning("선택하신 지역에 해당하는 월세 데이터가 존재하지 않습니다.")
else:
    col1, col2, col3, col4 = st.columns(4)
    
    median_py_price = sub_df['평당가격'].median() if '평당가격' in sub_df.columns else 0
    median_deposit = sub_df['보증금(만원)'].median() if '보증금(만원)' in sub_df.columns else 0
    avg_building_age = sub_df['건물나이'].mean() if '건물나이' in sub_df.columns else 0
    basement_cnt = (sub_df['지하여부_표시'] == '지하/반지하').sum() if '지하여부_표시' in sub_df.columns else 0
    basement_ratio = (basement_cnt / len(sub_df)) * 100 if len(sub_df) > 0 else 0

    with col1:
        st.metric(label="💵 평당 가격 (중앙값)", value=f"{median_py_price:,.1f} 만원")
    with col2:
        st.metric(label="🏦 보증금 (중앙값)", value=f"{median_deposit:,.0f} 만원")
    with col3:
        st.metric(label="🏗️ 평균 건물 나이", value=f"{avg_building_age:.1f} 년")
    with col4:
        st.metric(label="🏚️ 지하/반지하 비중", value=f"{basement_ratio:.1f}%", delta=f"총 {basement_cnt} 건")

    st.divider()

    st.markdown("### 🏷️ 가격 범위 (최저 ~ 최대)")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        min_dep = sub_df['보증금(만원)'].min()
        max_dep = sub_df['보증금(만원)'].max()
        st.info(f"**보증금 범위**\n\n최저 **{min_dep:,.0f}** 만원 ~ 최고 **{max_dep:,.0f}** 만원")
        
    with p_col2:
        min_rent = sub_df['임대료(만원)'].min()
        max_rent = sub_df['임대료(만원)'].max()
        st.success(f"**월세(임대료) 범위**\n\n최저 **{min_rent:,.0f}** 만원 ~ 최고 **{max_rent:,.0f}** 만원")
        
    with p_col3:
        min_py = sub_df['평당가격'].min()
        max_py = sub_df['평당가격'].max()
        st.warning(f"**평당가격 범위**\n\n최저 **{min_py:,.1f}** 만원 ~ 최고 **{max_py:,.1f}** 만원")

    st.divider()

    plot_df = sub_df 
    if selected_gu == "전체" and len(plot_df) > 10000:
        plot_df = plot_df.sample(n=10000, random_state=42)
        st.info("📌 서울시 전체는 데이터가 많아 그래프는 10,000건을 무작위 샘플링하여 표시합니다.")

    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.markdown("#### 📊 보증금 vs 월세 분포")
        fig_scatter = px.scatter(
            plot_df, 
            x="보증금(만원)", 
            y="임대료(만원)", 
            size="평수" if "평수" in sub_df.columns else None,
            color="지하여부_표시" if "지하여부_표시" in sub_df.columns else None,
            hover_data=["건물용도", "건물나이", "층"],
        )
        st.plotly_chart(fig_scatter, width='stretch')

    with c_col2:
        st.markdown("#### 🏢 건물 용도별 분포")
        purpose_cnt = sub_df["건물용도"].value_counts().reset_index(name="건수")
        purpose_cnt.columns = ["건물용도", "건수"]

        fig_pie = px.pie(
            purpose_cnt,
            names="건물용도",
            values="건수",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("📄 선택 지역 원본 상세 데이터 보기"):
        st.caption(f"총 {len(sub_df):,}건 중 상위 1,000건만 표시합니다.")
        st.dataframe(sub_df.head(1000), width=800, hide_index=True)
        csv = sub_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 전체 데이터 다운로드", data=csv, file_name="filtered_data.csv", mime="text/csv")
