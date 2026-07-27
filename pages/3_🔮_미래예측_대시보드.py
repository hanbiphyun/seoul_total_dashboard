import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import json

# ==========================================
# 1. 페이지 및 상태 초기화
# ==========================================
st.set_page_config(page_title="서울시 동네별 1년 뒤 시세 지도", layout="wide", page_icon="🏢")
st.title("🏢 서울시 AI 월세 지도 (1년 뒤 미래 시세 예측)")
st.markdown("과거 데이터를 바탕으로 딥러닝 모델이 예측한 **1년 뒤의 해당 동네 예상 평당가**를 보여줍니다.")

# ==========================================
# 2. 자산 로드
# ==========================================
@st.cache_resource
def load_assets():
    dong_mapping = joblib.load('ts_dong_mapping.pkl')
    # 팀원이 준 평가 성적표 파일 (lstm_pred가 들어있는 파일)
    test_df = pd.read_csv('test_dong_ts.csv') 
    
    with open('seoul_dong.geojson', 'r', encoding='utf-8') as f:
        seoul_geojson = json.load(f)
        
    return dong_mapping, test_df, seoul_geojson

dong_mapping, test_df, seoul_geojson = load_assets()

# ==========================================
# 3. 전체 동네 예측 결과 가져오기
# ==========================================
@st.cache_data
def get_all_predictions(dong_mapping, test_df):
    results = []
    
    if '계약일' in test_df.columns:
        test_df = test_df.sort_values(by='계약일')

    for dong, info in dong_mapping.items():
        dong_code = info['법정동코드']
        
        dong_df = test_df[test_df['법정동코드'] == dong_code]
        
        if dong_df.empty:
            continue
            
        pred_price = dong_df.iloc[-1]['lstm_pred']
        gu_code = dong_df.iloc[-1].get('자치구코드', dong_code // 100000)
        
        results.append({'dong_name': dong, 'gu_code': gu_code, 'predicted_price': pred_price})
        
    df = pd.DataFrame(results)
    if not df.empty:
        df['gu_avg_price'] = df.groupby('gu_code')['predicted_price'].transform('mean')
    
    return df

pred_df = get_all_predictions(dong_mapping, test_df)

# ==========================================
# 4. 데이터가 있는 동네만 필터링 & 초기화
# ==========================================
# 🌟 핵심: 예측 결과(pred_df)에 존재하는 동네만 뽑아서 리스트로 만듭니다.
valid_dongs = sorted(pred_df['dong_name'].tolist()) if not pred_df.empty else []

if 'selected_dong' not in st.session_state:
    if '역삼동' in valid_dongs:
        st.session_state['selected_dong'] = '역삼동'
    elif valid_dongs:
        st.session_state['selected_dong'] = valid_dongs[0]
    else:
        st.session_state['selected_dong'] = None

# ==========================================
# 5. UI: 사이드바 (조건 검색)
# ==========================================
st.sidebar.header("🔍 조건 검색")

if valid_dongs:
    sidebar_selection = st.sidebar.selectbox(
        "1년 뒤 시세를 확인할 동네 선택", 
        valid_dongs, # 🌟 모든 동네(dong_mapping)가 아닌 데이터가 있는 동네만 띄움
        index=valid_dongs.index(st.session_state['selected_dong']) if st.session_state['selected_dong'] in valid_dongs else 0
    )

    if sidebar_selection != st.session_state['selected_dong']:
        st.session_state['selected_dong'] = sidebar_selection
        st.rerun()

area = st.sidebar.number_input("내 방(또는 알아볼 방) 평수", min_value=2.0, max_value=50.0, value=10.0, step=1.0)
user_rent = st.sidebar.number_input("🤔 비교할 현재 월세 (만원)", min_value=0, value=60, step=5)

# ==========================================
# 6. UI: 메인 화면 (지도 및 상세 분석)
# ==========================================
col_map, col_details = st.columns([1.5, 1])

with col_map:
    st.subheader("🗺️ 서울시 동별 1년 뒤 예상 평당가 지도")
    
    curr_dong = st.session_state.get('selected_dong')
    
    if curr_dong and curr_dong in dong_mapping:
        lat = dong_mapping[curr_dong]['lat']
        lon = dong_mapping[curr_dong]['lon']
    else:
        lat, lon = 37.5665, 126.9780 
        
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
    
    if not pred_df.empty:
        folium.Choropleth(
            geo_data=seoul_geojson,
            data=pred_df,
            columns=['dong_name', 'predicted_price'],
            key_on='feature.properties.adm_nm', 
            fill_color='YlGn',
            fill_opacity=0.4,
            line_opacity=0.1,
            line_weight=0.5,
            legend_name='AI 예측 1년 뒤 평균 평당가 (만원)'
        ).add_to(m)

    folium.GeoJson(
        seoul_geojson,
        name="Click Layer",
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'lightblue', 'weight': 1, 'fillOpacity': 0},
        tooltip=folium.GeoJsonTooltip(fields=['adm_nm'], aliases=['동 이름:']),
    ).add_to(m)
    
    if curr_dong:
        folium.Marker(
            [lat, lon], 
            icon=folium.Icon(color='blue', icon='star'),
            tooltip=f"현재 선택: {curr_dong}"
        ).add_to(m)

    map_data = st_folium(m, width=700, height=500, returned_objects=["last_active_drawing"])
    
    if map_data["last_active_drawing"]:
        clicked_dong = map_data["last_active_drawing"]["properties"]["adm_nm"]
        # 🌟 지도에서 클릭한 동네가 유효한 데이터가 있을 때만 반응하도록 수정
        if clicked_dong in valid_dongs and clicked_dong != st.session_state.get('selected_dong'):
            st.session_state['selected_dong'] = clicked_dong
            st.rerun()

# ==========================================
# 7. UI: 우측 상세 분석 (즉석 비교)
# ==========================================
with col_details:
    st.subheader(f"📊 {curr_dong if curr_dong else '선택된 동네'} 1년 뒤 미래 시세 분석")
    
    if pred_df.empty or not curr_dong:
        st.warning("⚠️ 시세 예측 데이터가 없습니다.")
    else:
        # 🌟 이제 에러 경고문을 띄울 필요가 없음 (유효한 동네만 넘어오기 때문)
        dong_data = pred_df[pred_df['dong_name'] == curr_dong].iloc[0]
        predicted_pyeong_price = dong_data['predicted_price']
        gu_avg_price = dong_data['gu_avg_price']
        expected_rent = int(predicted_pyeong_price * area)
        
        st.markdown("### 🏘️ 1년 뒤 동네 상대 비교")
        diff_gu = predicted_pyeong_price - gu_avg_price
        gu_diff_pct = (diff_gu / gu_avg_price) * 100
        
        st.metric(label=f"AI 예측 {curr_dong} 1년 뒤 평당가", value=f"{predicted_pyeong_price:.1f} 만원/평", 
                  delta=f"같은 구(주변) 1년 뒤 평균 대비 {abs(gu_diff_pct):.1f}% {'비쌈' if diff_gu > 0 else '저렴'}",
                  delta_color="inverse" if diff_gu > 0 else "normal")
        
        if diff_gu > 0:
            st.info(f"💡 1년 뒤 {curr_dong}은 주변 동네들 평균({gu_avg_price:.1f}만원)보다 시세가 높게 형성될 전망입니다.")
        else:
            st.success(f"💡 1년 뒤 {curr_dong}은 주변 동네들 평균({gu_avg_price:.1f}만원)보다 저렴하여 가성비가 좋을 전망입니다.")
            
        st.divider()
        
        st.markdown(f"### 🛏️ {int(area)}평 현재가 vs 1년 뒤 예측가 비교")
        diff_rent = expected_rent - user_rent
        rent_diff_pct = (diff_rent / user_rent) * 100 if user_rent > 0 else 0
        
        st.metric(label="딥러닝 예상 1년 뒤 적정 월세", value=f"{expected_rent} 만원")
        
        if diff_rent > 0:
            st.error(f"📈 지금 알아보신 {user_rent}만원 방은 1년 뒤에 **{diff_rent}만원 오를 전망입니다** (+{rent_diff_pct:.1f}%)")
        else:
            st.success(f"📉 지금 알아보신 {user_rent}만원 방은 1년 뒤에 **{abs(diff_rent)}만원 저렴해질 전망입니다** ({rent_diff_pct:.1f}%)")
