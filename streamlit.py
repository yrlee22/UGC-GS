import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import mapping
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

# 초기 설정
st.set_page_config(layout="wide", page_title="서울시 지하공사 취약 지역 리스크 평가 - 지반침하 위험 중심으로")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

@st.cache_resource
def get_map_center():
    return [37.5665, 126.9780], 11  # 서울 시청 기준

@st.cache_data
def load_data():
    df = pd.read_csv("df_test_with_proba.csv")
    gdf = gpd.read_file("gdf_final.geojson")
    shap_df = pd.read_csv("group_shap.csv")
    detail_df = pd.read_csv("dong_to_shap.csv")
    return df, gdf, shap_df, detail_df

df, gdf, shap_df, detail_df = load_data()

def get_grade(prob):
    if prob > 0.75:
        return "1등급"
    elif prob > 0.55:
        return "2등급"
    elif prob > 0.35:
        return "3등급"
    elif prob > 0.15:
        return "4등급"
    else:
        return "5등급"

# 자치구 및 법정동 선택
st.subheader("서울시 지하공사 취약 지역 리스크 평가 - 지반침하 위험 중심으로")
gu_list = sorted(df["시군구명"].unique())
col1, col2 = st.columns(2)

with col1:
    selected_gu = st.selectbox("자치구 선택", ["선택 안 함"] + gu_list)

with col2:
    if selected_gu != "선택 안 함":
        dong_list = sorted(df[df["시군구명"] == selected_gu]["법정동명"].unique())
        selected_dong = st.selectbox("법정동 선택", ["선택 안 함"] + dong_list)
    else:
        selected_dong = None

selected_full = f"{selected_gu}_{selected_dong}" if selected_gu != "선택 안 함" and selected_dong != "선택 안 함" else None

# 지도 시각화
left, right = st.columns([7.5, 2.5], gap="small")

with left:
    center, zoom = get_map_center()
    m = folium.Map(location=center, zoom_start=zoom)

    for _, row in gdf.iterrows():
        name = row["법정동"]
        proba = row["예측확률"]

        # 색상 지정
        if proba > 0.75:
            fill_color = "darkred"
        elif proba > 0.55:
            fill_color = "red"
        elif proba > 0.35:
            fill_color = "orange"
        elif proba > 0.15:
            fill_color = "#FFDA7A"
        else:
            fill_color = "#A9A9A9"

        is_selected = selected_full == name
        current_opacity = 1.0 if is_selected else 0.5
        border_color = fill_color
        border_weight = 4 if is_selected else 1

        # 팝업 HTML
        score_text = f"{round(proba * 100, 1)}%"
        grade_text = get_grade(proba)
        popup_html = f"""
            <div style="font-size: 12px;">
                <b>{name.replace('_', ' ')}</b><br>
                <b>위험률:</b> {score_text}<br>
                <b>위험 등급:</b> {grade_text}
            </div>
        """

        # GeoJSON geometry
        feature = {
            "type": "Feature",
            "properties": {"name": name},
            "geometry": mapping(row["geometry"])
        }

        style_dict = {
            'fillColor': fill_color,
            'color': border_color,
            'weight': border_weight,
            'fillOpacity': current_opacity,
            'opacity': 1.0
        }

        g = folium.GeoJson(
            data=feature,
            name=name,
            tooltip=f"<div style='font-size:12px'>{name.replace('_', ' ')}</div>",
            style_function=lambda x, sd=style_dict: sd,
            highlight_function=lambda x: {}
        )
        g.add_child(folium.Popup(popup_html, max_width=250))
        g.add_to(m)

        if is_selected:
            centroid = row["geometry"].centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=""),
                popup=folium.Popup(popup_html, max_width=250, show=True)
            ).add_to(m)

    # 클릭된 도형 정보 가져오기
    st_data = st_folium(m, use_container_width=True, height=600, returned_objects=["last_active_drawing"])
    clicked_name = None
    if st_data and isinstance(st_data.get("last_active_drawing"), dict):
        clicked_name = st_data["last_active_drawing"].get("properties", {}).get("name")

# SHAP 요인 시각화
with right:
    display_dong = clicked_name if clicked_name else selected_full

    if display_dong is None:
        st.info("자치구와 법정동을 선택해주세요.")
    else:
        st.markdown(f"##### {display_dong.replace('_', ' ')} 지반침하 위험 요인")

        display_proba = gdf[gdf["법정동"] == display_dong]["예측확률"].values
        if display_proba.size > 0 and not np.isnan(display_proba[0]):
            pct = round(display_proba[0] * 100, 1)
            grade = get_grade(display_proba[0])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**위험률:** {pct}%")
            with col_b:
                st.markdown(f"**위험 등급:** {grade}")
        else:
            st.markdown("**위험 정보 없음**")

        dong_risk = shap_df[shap_df["법정동"] == display_dong]
        if dong_risk.empty:
            st.warning("선택된 지역의 위험 요인 분석 정보가 없습니다.")
        else:
            plot_df = dong_risk.sort_values("shap_value", ascending=False).copy()
            plot_df["요인라벨"] = plot_df["위험요인그룹"].replace({
                r"지하 공사": "지하공사",
                r"지하시설물": "지하시설물"
            }, regex=True)

            colors = ["#E74C3C" if val > 0 else "#3498DB" for val in plot_df["shap_value"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=plot_df["요인라벨"],
                y=plot_df["shap_value"],
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>SHAP 기여도: %{y:.2f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis=dict(tickfont=dict(family="Malgun Gothic", size=14)),
                yaxis=dict(title=None, tickfont=dict(family="Malgun Gothic", size=14)),
                
                font=dict(family="Malgun Gothic", size=14),
                height=280,
                template="plotly_white",
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            
            st.markdown("""
                <style>
                .block-container > div:has(.js-plotly-plot) + div {
                    margin-top: -20px !important;
                }
                </style>
            """, unsafe_allow_html=True)

            merged_df = detail_df.merge(plot_df[["위험요인그룹"]], on="위험요인그룹", how="inner")
            merged_df = merged_df[merged_df["법정동"] == display_dong]

            if not merged_df.empty:
                top_5 = merged_df.sort_values("shap_value", ascending=False).head(5)
                top_5_display = top_5[["위험요인그룹", "영향 인자", "shap_value"]].copy()
                top_5_display.columns = ["요인 분류", "세부 항목", "기여도"]
                top_5_display["세부 항목"] = top_5_display["세부 항목"].str.replace("_", " ")
                top_5_display["기여도"] = top_5_display["기여도"].map("{:.2f}".format)
                top_5_display = top_5_display.reset_index(drop=True)
                st.table(top_5_display)
