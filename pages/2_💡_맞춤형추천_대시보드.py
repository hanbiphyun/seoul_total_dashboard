import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import platform
import gdown  # 구글 드라이브 다운로드를 위해 추가됨

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title='서울 월세 거주지역 추천',
    page_icon='💡',
    layout='wide'
)

st.title('💡 서울 월세 거주지역 맞춤형 추천')
st.write(
    '2023~2025년 서울 월세 실거래 데이터를 바탕으로 '
    '보증금, 평당 임대료, 평수를 고려하여 '
    '거주 가능한 법정동을 추천합니다.'
)

# ==========================================
# ⭐️ 이 부분이 구글 드라이브에서 다운로드하도록 변경되었습니다!
# ==========================================
@st.cache_data
def load_data():
    # 아래 따옴표 안에 아까 구글 드라이브에서 복사한 파일 ID를 붙여넣으세요!
    file_id = "1DRpgms4olDIK4946bi4b79huYlcmQIm6"  
    
    local_path = "recommend_data.parquet"
    
    # 파일이 없거나 용량이 너무 작으면(오류 파일) 구글 드라이브에서 다운로드
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 100000:
        with st.spinner("서버에서 추천 데이터를 준비하고 있습니다. 잠시만 기다려주세요..."):
            gdown.download(id=file_id, output=local_path, quiet=False)
        
    data = pd.read_parquet(local_path)
    return data

try:
    df = load_data()
except Exception as error:
    st.error(f'데이터 불러오기 실패: 구글 드라이브 공유 권한이 "링크가 있는 모든 사용자"인지, 파일 ID가 정확한지 확인해주세요. (에러내용: {error})')
    st.stop()

def minmax_score(series, reverse=False):
    series = pd.to_numeric(series, errors='coerce')
    if series.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)

    series = series.fillna(series.median())
    min_value = series.min()
    max_value = series.max()

    if min_value == max_value:
        score = pd.Series(1.0, index=series.index)
    else:
        score = (series - min_value) / (max_value - min_value)

    if reverse:
        score = 1 - score
    return score

def filter_houses(
    data, deposit_min, deposit_max, price_min, price_max, 
    area_min, area_max, exclude_basement=False, building_types=None
):
    filtered = data[
        data['보증금(만원)'].between(deposit_min, deposit_max) &
        data['평당가격'].between(price_min, price_max) &
        data['평수'].between(area_min, area_max)
    ].copy()

    if exclude_basement and '지하여부' in filtered.columns:
        filtered = filtered[filtered['지하여부'].isna() | (filtered['지하여부'] == 0)].copy()

    if building_types is not None and len(building_types) > 0 and '건물용도' in filtered.columns:
        filtered = filtered[filtered['건물용도'].isin(building_types)].copy()

    return filtered

def aggregate_by_region(filtered):
    if filtered.empty:
        return pd.DataFrame()

    summary = (
        filtered.groupby(['자치구명', '법정동명'], as_index=False)
        .agg(
            조건충족_거래수=('법정동명', 'size'),
            보증금_중앙값=('보증금(만원)', 'median'),
            평당가격_중앙값=('평당가격', 'median'),
            평당가격_평균=('평당가격', 'mean'),
            평당가격_표준편차=('평당가격', 'std'),
            월세_중앙값=('임대료(만원)', 'median'),
            평수_중앙값=('평수', 'median'),
            건물나이_중앙값=('건물나이', 'median'),
            거래연도수=('계약연도', 'nunique')
        )
    )

    summary['가격변동계수'] = summary['평당가격_표준편차'] / summary['평당가격_평균']
    summary['가격변동계수'] = summary['가격변동계수'].replace([np.inf, -np.inf], np.nan)
    return summary

def add_fit_rate(summary, full_data):
    total_counts = full_data.groupby(['자치구명', '법정동명']).size().reset_index(name='지역전체_거래수')
    result = summary.merge(total_counts, on=['자치구명', '법정동명'], how='left')
    result['조건적합률'] = result['조건충족_거래수'] / result['지역전체_거래수']
    return result

def calculate_score(summary, min_transactions):
    result = summary[summary['조건충족_거래수'] >= min_transactions].copy()
    if result.empty:
        return result

    result['보증금점수'] = minmax_score(result['보증금_중앙값'], reverse=True)
    result['평당가격점수'] = minmax_score(result['평당가격_중앙값'], reverse=True)
    result['거래량점수'] = minmax_score(np.log1p(result['조건충족_거래수']))
    result['거래지속성점수'] = result['거래연도수'].clip(lower=1, upper=3) / 3
    result['가격안정성점수'] = minmax_score(result['가격변동계수'], reverse=True).fillna(0.5)

    result['거래안정성점수'] = (
        result['거래량점수'] * 0.50 +
        result['거래지속성점수'] * 0.30 +
        result['가격안정성점수'] * 0.20
    )

    result['평수점수'] = minmax_score(result['평수_중앙값'])
    result['건물상태점수'] = minmax_score(result['건물나이_중앙값'], reverse=True).fillna(0.5)
    result['조건적합률점수'] = minmax_score(result['조건적합률'])

    result['추천점수'] = (
        result['보증금점수'] * 0.20 +
        result['평당가격점수'] * 0.20 +
        result['거래안정성점수'] * 0.30 +
        result['평수점수'] * 0.15 +
        result['건물상태점수'] * 0.10 +
        result['조건적합률점수'] * 0.05
    ) * 100

    result['추천지역'] = result['자치구명'].astype(str) + ' ' + result['법정동명'].astype(str)
    result = result.sort_values(by=['추천점수', '조건충족_거래수'], ascending=[False, False]).reset_index(drop=True)
    result['추천순위'] = np.arange(1, len(result) + 1)
    
    return result

def make_reason(row):
    reasons = []
    if row['보증금점수'] >= 0.7: reasons.append('보증금 부담이 낮은 편')
    if row['평당가격점수'] >= 0.7: reasons.append('평당 임대료가 낮은 편')
    if row['거래안정성점수'] >= 0.7: reasons.append('조건에 맞는 거래가 안정적으로 존재')
    if row['평수점수'] >= 0.7: reasons.append('상대적으로 넓은 주거 면적')
    if row['건물상태점수'] >= 0.7: reasons.append('건물 연식이 비교적 최근')
    if not reasons: reasons.append('가격과 거래 안정성의 균형이 좋음')
    return ', '.join(reasons)

def run_recommendation(
    data, deposit_min, deposit_max, price_min, price_max, 
    area_min, area_max, exclude_basement, building_types, min_transactions, top_n
):
    relaxation_rates = [0.0, 0.1, 0.2, 0.3]
    for rate in relaxation_rates:
        filtered = filter_houses(
            data=data,
            deposit_min=max(0, deposit_min * (1 - rate)),
            deposit_max=(deposit_max * (1 + rate)),
            price_min=max(0, price_min * (1 - rate)),
            price_max=(price_max * (1 + rate)),
            area_min=max(0, area_min * (1 - rate)),
            area_max=(area_max * (1 + rate)),
            exclude_basement=exclude_basement,
            building_types=building_types
        )

        if filtered.empty:
            continue

        summary = aggregate_by_region(filtered)
        summary = add_fit_rate(summary, data)
        scored = calculate_score(summary, min_transactions)

        if not scored.empty:
            scored['추천이유'] = scored.apply(make_reason, axis=1)
            return scored.head(top_n), filtered, rate

    return pd.DataFrame(), pd.DataFrame(), None

st.sidebar.header('🔍 희망 조건 입력')

deposit_min = st.sidebar.number_input('최소 보증금(만원)', min_value=0, value=500, step=100)
deposit_max = st.sidebar.number_input('최대 보증금(만원)', min_value=0, value=5000, step=100)
price_min = st.sidebar.number_input('최소 평당 임대료(만원/평)', min_value=0.0, value=2.0, step=0.5)
price_max = st.sidebar.number_input('최대 평당 임대료(만원/평)', min_value=0.0, value=8.0, step=0.5)
area_min = st.sidebar.number_input('최소 평수', min_value=0.0, value=5.0, step=0.5)
area_max = st.sidebar.number_input('최대 평수', min_value=0.0, value=12.0, step=0.5)

exclude_basement = st.sidebar.checkbox('지하 매물 제외', value=True)
min_transactions = st.sidebar.slider('지역별 최소 거래 수', min_value=1, max_value=30, value=3)
top_n = st.sidebar.slider('추천 지역 수', min_value=5, max_value=20, value=10)

if '건물용도' in df.columns:
    building_options = sorted(df['건물용도'].dropna().astype(str).unique().tolist())
    building_types = st.sidebar.multiselect('건물용도 선택', options=building_options, default=[])
else:
    building_types = None

if st.sidebar.button('🚀 추천 지역 찾기', type='primary'):
    if deposit_min > deposit_max:
        st.error('최소 보증금은 최대 보증금보다 작아야 합니다.')
    elif price_min > price_max:
        st.error('최소 평당 임대료는 최대 평당 임대료보다 작아야 합니다.')
    elif area_min > area_max:
        st.error('최소 평수는 최대 평수보다 작아야 합니다.')
    else:
        with st.spinner("AI가 최적의 매물을 찾고 있습니다..."):
            regions, matched, relaxation_rate = run_recommendation(
                data=df, deposit_min=deposit_min, deposit_max=deposit_max,
                price_min=price_min, price_max=price_max, area_min=area_min,
                area_max=area_max, exclude_basement=exclude_basement,
                building_types=building_types, min_transactions=min_transactions, top_n=top_n
            )

        if regions.empty:
            st.warning('조건을 30%까지 완화했지만 추천 가능한 지역이 없습니다. 조건을 조금 더 여유롭게 설정해 보세요.')
        else:
            if relaxation_rate == 0:
                st.success('입력 조건을 만족하는 지역을 찾았습니다.')
            else:
                st.warning(f'정확한 조건의 결과가 부족하여 조건을 {int(relaxation_rate * 100)}% 완화하여 추천해 드립니다.')

            # 요약 지표
            col1, col2, col3 = st.columns(3)
            col1.metric('📌 조건 충족 거래', f'{len(matched):,}건')
            col2.metric('🏆 추천 지역', f'{len(regions):,}곳')
            col3.metric('⭐ 최고 추천점수', f'{regions["추천점수"].max():.1f}점')

            # 추천 결과표
            st.subheader('📑 상세 추천 지역 리스트')
            display_columns = [
                '추천순위', '추천지역', '추천점수', '조건충족_거래수', '조건적합률',
                '보증금_중앙값', '평당가격_중앙값', '월세_중앙값', '평수_중앙값', '건물나이_중앙값', '추천이유'
            ]
            display_df = regions[display_columns].copy()
            display_df['조건적합률'] = display_df['조건적합률'] * 100
            display_df = display_df.rename(columns={'조건적합률': '조건적합률(%)'})

            st.dataframe(display_df.round(2), width='stretch', hide_index=True)

            # 추천 점수 그래프
            st.divider()
            st.subheader('📊 추천 점수 및 거래 수 비교')
            
            # 레이아웃을 2단으로 나누어 그래프 배치
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                plot_df = regions.sort_values('추천점수', ascending=True).copy()
                fig, ax = plt.subplots(figsize=(6, 5))
                bars = ax.barh(plot_df['추천지역'], plot_df['추천점수'], color='cornflowerblue')
                ax.set_xlabel('추천 점수')
                ax.set_title('사용자 조건 기반 추천 점수')
                for bar, value in zip(bars, plot_df['추천점수']):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f'{value:.1f}점', va='center')
                ax.set_xlim(0, max(105, plot_df['추천점수'].max() + 8))
                fig.tight_layout()
                st.pyplot(fig)

            with g_col2:
                transaction_df = regions.sort_values('조건충족_거래수', ascending=True).copy()
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                bars2 = ax2.barh(transaction_df['추천지역'], transaction_df['조건충족_거래수'], color='lightgreen')
                ax2.set_xlabel('거래 수')
                ax2.set_title('추천 지역별 조건 충족 거래 수')
                for bar, value in zip(bars2, transaction_df['조건충족_거래수']):
                    ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {int(value):,}건', va='center')
                fig2.tight_layout()
                st.pyplot(fig2)

            # 개별 거래 확인
            with st.expander('📄 조건을 만족한 개별 거래 원본 데이터 보기 (최대 1000건)'):
                house_columns = [col for col in ['계약연도', '자치구명', '법정동명', '보증금(만원)', '임대료(만원)', '평당가격', '평수', '건물나이', '지하여부', '건물용도'] if col in matched.columns]
                matched_display = matched[house_columns].sort_values(by=['보증금(만원)', '평당가격']).head(1000)
                st.dataframe(matched_display,width='stretch', hide_index=True)
