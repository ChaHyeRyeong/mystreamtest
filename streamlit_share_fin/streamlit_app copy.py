import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import math
from datetime import datetime
import re
import html
from pathlib import Path
import streamlit.components.v1 as components
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
from urllib.parse import quote, urlencode


timeline_component = components.declare_component(
    "course_timeline",
    path=str(
        Path(__file__).parent
        / "components"
        / "timeline"
    ),
)

from utils.data_loader import (
    load_all_resources,
    build_place_info_lookup
)

from utils.predict_congestion import (
    get_weather_multiplier_detailed,
    get_congestion_level,
    predict_congestion_final,
    get_daily_peak_congestion,
    get_auto_weather,
    load_weather_data
)

# 유튜브용 API
api_key = st.secrets.get("YOUTUBE_API_KEY", "")


# ---------------------------------
# YouTube API 영상 검색 함수
# ---------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def search_videos(api_key, keyword):
    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key,
        cache_discovery=False
    )

    response = youtube.search().list(
        part="snippet",
        q=keyword,
        type="video",
        maxResults=3,
        order="relevance",
        regionCode="KR",
        relevanceLanguage="ko",
        videoEmbeddable="true",
        safeSearch="moderate"
    ).execute()

    videos = []

    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")

        if not video_id:
            continue

        snippet = item.get("snippet", {})

        videos.append({
            "video_id": video_id,
            "title": snippet.get("title", "제목 없음"),
            "channel": snippet.get("channelTitle", "채널 정보 없음"),
            "description": snippet.get("description", "")
        })

    return videos


# 페이지 설정
st.set_page_config(page_title="강원도 틈새여행", page_icon="🌿", layout="wide")

# 왼쪽 페이지 목록 숨기기
st.markdown("""
    <style>
        /* 왼쪽 사이드바 전체 숨기기 */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* 사이드바 접기/펼치기 버튼 숨기기 */
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }

        /* 메인 콘텐츠 영역 확장 */
        [data-testid="stMain"] {
            margin-left: 0;
        }
    </style>
""", unsafe_allow_html=True)
# ==========================================
# CSS 스타일 로드
# ==========================================
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent / "style.css"

with open(CSS_PATH, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# 혼잡도 등급별 파스텔 색 (배경, 글자, 그래프 막대)
GRADE_STYLE = {
    "여유": {"emoji": "🌿", "bg": "#D8F1E3", "fg": "#2F7A55", "bar": "#8FD1AE"},
    "보통": {"emoji": "🌼", "bg": "#FFF3C4", "fg": "#8A6A12", "bar": "#F5D97A"},
    "주의": {"emoji": "🍊", "bg": "#FFE0C7", "fg": "#A2541C", "bar": "#F7B98A"},
    "혼잡": {"emoji": "🔥", "bg": "#FBD5D5", "fg": "#A33A3A", "bar": "#EE9A9A"},
}
CLOSED_STYLE = {"emoji": "🌙", "bg": "#EEF0EF", "fg": "#7A8A83", "bar": "#D5DBD8"}

WEATHER_ICON = {"맑음": "☀️", "비": "🌧️", "눈": "❄️", "폭염": "🥵", "한파": "🥶"}

# ==========================================
# esc
# ==========================================

def esc(value):
    return html.escape(str(value))


# ==========================================
# 세션 상태
# ==========================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "혼잡도 검색"

if "show_home_test" not in st.session_state:
    st.session_state.show_home_test = False

if "my_course" not in st.session_state:
    st.session_state.my_course = []

if "course_schedule" not in st.session_state:
    st.session_state.course_schedule = {}

if "show_course_detail" not in st.session_state:
    st.session_state.show_course_detail = False

MAX_COURSE_PLACES = 5

# ==========================================
# 상단 배너 + 메뉴
# ==========================================
st.markdown(
    """
    <div class="hero">
        <span class="hero-eyebrow">🌿 붐비지 않는 강원도를 찾아서</span>
        <div class="hero-title">강원도 틈새여행</div>
        <p class="hero-sub">날짜를 고르면 한적한 관광지와 그날의 예상 날씨를 함께 알려 드려요.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.radio(
    "메뉴",
    ["홈", "혼잡도 검색"],
    horizontal=True,
    key="active_page",
    label_visibility="collapsed",
)


# st.divider()

# ==========================================
# 0. 데이터 및 모델 로드
# ==========================================

(
    df_meta,
    indoor_status_map,
    model,
    place_monthly_weights,
    profile_lookup
) = load_all_resources()


# ==========================================
# 일정 추가 모달 함수 추가
# ==========================================


@st.dialog("🧭 My Course 일정 추가")
def add_place_dialog(
    place_name,
    visit_date,
    weather_condition,
    precipitation,
    hourly_scores=None,
):
    st.write(f"**관광지:** {place_name}")
    st.caption(f"방문 날짜: {visit_date}")

    with st.form(f"course_form_{place_name}"):

        start_hour = st.selectbox(
            "방문 시작 시간",
            options=list(range(24)),
            index=10,
            format_func=lambda h: f"{h:02d}:00"
        )

        duration = st.selectbox(
            "예상 체류시간",
            options=[1, 2, 3, 4],
            format_func=lambda h: f"{h}시간"
        )

        submitted = st.form_submit_button(
            "코스에 추가",
            use_container_width=True,
            type="primary"
        )

    
    if submitted:
        course = st.session_state.my_course
        schedule = st.session_state.course_schedule

        # 기존 코스의 날짜 확인
        existing_dates = {
            item["date"]
            for item in schedule.values()
            if item.get("date")
        }

        # 하루 일정이므로 날짜가 다르면 추가 차단
        if existing_dates and visit_date not in existing_dates:
            st.error(
                f"현재 코스는 {next(iter(existing_dates))} 일정입니다. "
                f"같은 날짜의 관광지만 추가할 수 있어요."
            )
            return
    
        # 새 일정이 차지하는 시간대
        new_hours = {
            (start_hour + offset) % 24
            for offset in range(duration)
        }

        # 기존 관광지와 시간 중복 확인
        conflicts = []

        for existing_place in course:
            if existing_place == place_name:
                continue

            existing_item = schedule.get(existing_place)

            if not existing_item:
                continue

            existing_start = existing_item["start_hour"]
            existing_duration = existing_item["duration"]

            existing_hours = {
                (existing_start + offset) % 24
                for offset in range(existing_duration)
            }

            if new_hours & existing_hours:
                conflicts.append(existing_place)

        if conflicts:
            st.error(
                "선택한 시간에 이미 일정이 있습니다: "
                + ", ".join(conflicts)
                + "\n다른 시작 시간으로 선택해 주세요."
            )

        else:
            if place_name not in course:
                if len(course) >= MAX_COURSE_PLACES:
                    st.error("관광지는 최대 5개까지 추가할 수 있습니다.")
                    return

                course.append(place_name)


            # 시작시간 기준 혼잡도 점수 계산
            scores = hourly_scores or []

            start_score = None

            for row in scores:
                if int(row.get("hour", -1)) == int(start_hour):
                    if row.get("final_score") is not None:
                        start_score = float(row["final_score"])
                    break

            # 혼잡도 점수에 따른 등급 계산
            if start_score is not None:
                if start_score <= 39:
                    start_level = "여유"
                elif start_score <= 69:
                    start_level = "보통"
                elif start_score <= 84:
                    start_level = "주의"
                else:
                    start_level = "혼잡"
            else:
                start_level = None
                        
            st.session_state.course_schedule[place_name] = {
                "date": visit_date,
                "start_hour": start_hour,
                "duration": duration,
                "weather_condition": weather_condition,
                "precipitation": precipitation,
                "congestion_score": start_score,
                "congestion_level": start_level,
            }

            st.session_state.show_course_detail = False
            st.rerun()


# ==========================================
# 5. 메인 화면 검색 필터 + 검색 버튼
# ==========================================
if st.session_state.active_page == "홈":
    st.markdown("## 🏠 홈 화면")
    st.info("홈 화면 테스트 영역입니다.")

else:
    region_col = (
        "지역" if "지역" in df_meta.columns else df_meta.columns[3]
    )
    place_col_name = (
        "관광지명" if "관광지명" in df_meta.columns
        else ("관광지" if "관광지" in df_meta.columns else df_meta.columns[1])
    )
    place_info_lookup = build_place_info_lookup(
        df_meta, place_col_name
    )
    regions = sorted(df_meta[region_col].dropna().unique().tolist())

    # 검색 상태 초기화
    if "has_searched" not in st.session_state:
        st.session_state.has_searched = False

    if "search_params" not in st.session_state:
        st.session_state.search_params = None

    # form 내부에서는 검색 버튼을 누를 때까지 입력 변경이 제출되지 않음
    # with st.form("congestion_search_form"):
    with st.container(key="search_box"):
        st.markdown('<div class="section-title">🔍 어디로, 언제 떠나세요?</div>', unsafe_allow_html=True)

        with st.form("congestion_search_form", border=False):
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                selected_region_input = st.selectbox(
                    "지역 선택",
                    ["전체"] + regions,
                    key="region_input"
                )

            with filter_col2:
                selected_date_input = st.date_input(
                    "여행 날짜",
                    value=datetime.today(),
                    key="date_input"
                )

            with filter_col3:
                weather_option_input = st.selectbox(
                    "날씨 선택",
                    ["자동", "맑음", "비/눈", "폭염/한파"],
                    key="weather_input"
                )

            with filter_col4:
                congestion_filter_input = st.selectbox(
                    "혼잡도 필터",
                    [
                        "전체",
                        "여유 (0~39점)",
                        "보통 (40~69점)",
                        "주의 (70~84점)",
                        "혼잡 (85~100점)"
                    ],
                    key="congestion_input"
                )

            search_clicked = st.form_submit_button(
                "틈새 여행지 찾기",
                icon=":material/travel_explore:",
                use_container_width=True,
                type="primary"
            )

    # 검색 버튼을 누른 경우에만 검색 조건 저장
    if search_clicked:
        st.session_state.result_page = 1 # 새 검색 시 첫 페이지로 이동
        st.session_state.search_params = {
            "selected_region": selected_region_input,
            "selected_date": selected_date_input,
            "weather_option": weather_option_input,
            "congestion_filter": congestion_filter_input
        }
        st.session_state.has_searched = True

    #st.space("small")
    # st.divider()

    # 제출된 검색 조건이 있을 때만 결과 계산에 사용할 변수 설정
    if st.session_state.has_searched:
        params = st.session_state.search_params

        selected_region = params["selected_region"]
        selected_date = params["selected_date"]
        target_date_str = selected_date.strftime("%Y-%m-%d")
        weather_option = params["weather_option"]
        congestion_filter = params["congestion_filter"]

        # 지역별 관광지 목록
        if selected_region == "전체":
            filtered_places = (
                df_meta[place_col_name].dropna().tolist()
            )
        else:
            filtered_places = (
                df_meta.loc[
                    df_meta[region_col] == selected_region,
                    place_col_name
                ].dropna().tolist()
            )

        # 날씨 조건 변환
        if weather_option == "자동":
            weather_condition = "자동"
            precipitation = 0.0
        elif weather_option == "맑음":
            weather_condition = "맑음"
            precipitation = 0.0
        elif weather_option == "비/눈":
            weather_condition = "비"
            precipitation = 5.0
        elif weather_option == "폭염/한파":
            weather_condition = "폭염"
            precipitation = 0.0


    def weather_cards_html(region_names, target_date_str):
        """선택 지역의 예상 날씨 카드 (날씨 CSV의 지역·월일 기준 평년 예측)"""
        df = load_weather_data()
        month_day = datetime.strptime(target_date_str, "%Y-%m-%d").strftime("%m-%d")
        rows = df[df["월일_키"] == month_day].drop_duplicates("지역_키").set_index("지역_키")

        cards = []
        for region in region_names:
            key = str(region).strip()[:2]
            if key not in rows.index:
                continue
            row = rows.loc[key]
            weather = row["예측"]
            rain = float(row["비 확률"])
            snow = float(row["눈 발생 빈도"])
            cards.append(
                {
                    "region": region,
                    "weather": weather,
                    "icon": WEATHER_ICON.get(weather, "🌤️"),
                    "meta": f"비 확률 {rain:.0%} · 눈 {snow:.0%}",
                }
            )

        if not cards:
            return ""

        if len(cards) == 1:
            c = cards[0]
            return (
                '<div class="weather-wrap" style="grid-template-columns:1fr">'
                '<div class="weather-card big">'
                f'<div class="weather-icon">{c["icon"]}</div>'
                "<div>"
                f'<div class="weather-region">{esc(c["region"])} · {target_date_str} 예상 날씨</div>'
                f'<div class="weather-main">{esc(c["weather"])}</div>'
                f'<div class="weather-meta">{c["meta"]} · 평년 기록 기반 예측</div>'
                "</div></div></div>"
            )

        items = "".join(
            '<div class="weather-card">'
            f'<div class="weather-region">{esc(c["region"])}</div>'
            f'<div class="weather-main">{c["icon"]} {esc(c["weather"])}</div>'
            f'<div class="weather-meta">{c["meta"]}</div>'
            "</div>"
            for c in cards
        )
        return f'<div class="weather-wrap">{items}</div>'


    def hourly_chart(hourly_scores, bar_color):
        """시간대별 혼잡도 미니 막대그래프 (0~23시)"""
        hourly_df = pd.DataFrame(hourly_scores)
        chart_df = pd.DataFrame({"hour": range(24)}).merge(
            hourly_df[["hour", "final_score"]],
            on="hour",
            how="left"
        )

        fig = go.Figure(
            go.Bar(
                x=chart_df["hour"],
                y=chart_df["final_score"],
                marker_color=bar_color,
                marker_line_width=0,
                hovertemplate="%{x}시 · 혼잡도 %{y:.1f}점<extra></extra>"
            )
        )
        fig.update_layout(
            height=96,
            margin=dict(l=0, r=0, t=4, b=14),
            xaxis=dict(
                range=[-0.5, 23.5],
                tickmode="array",
                tickvals=list(range(0, 24, 3)),
                ticktext=[f"{h}시" for h in range(0, 24, 3)],
                tickfont=dict(size=9, color="#9AA8A2"),
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                range=[0, 100],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            showlegend=False,
            bargap=0.2,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig


    # ==========================================
    # 6. 결과 출력 대시보드
    #    왼쪽: 검색 결과
    #    오른쪽: My Course
    # ==========================================

    result_col, course_col = st.columns([1.8, 1], gap="large")

    # ==========================================
    # 왼쪽: 추천 관광지
    # ==========================================
    with result_col:

        if st.session_state.show_home_test:
            st.markdown("## 🏠 홈 화면")
            st.info("홈 버튼 클릭 테스트 화면입니다.")


        with st.expander("ℹ️ 혼잡도 산출 방식 및 이용 시 주의사항"):
            st.markdown("""
            **1. 혼잡도 산출 방식**

            과거 혼잡도 데이터를 바탕으로 예측 모델을 통해
            선택한 날짜와 시간대의 예상 혼잡도를 산출합니다.
            최종 점수에는 서비스에 적용된 보정 기준이 반영될 수 있습니다.

            **2. 점수 및 등급 해석**

            혼잡도 점수는 관광지별 상대적인 혼잡 수준을 나타내는 지표이며,
            실제 방문객 수나 대기 시간을 직접 의미하지 않습니다.

            **3. 이용 시 주의사항**

            - 예측값은 실제 현장 상황과 다를 수 있습니다.
            - 행사, 날씨, 교통 등 외부 요인에 따라 혼잡도가 달라질 수 있습니다.
            - 운영시간 외의 예측값은 낮은 혼잡도를 의미하지 않습니다.
            - 여행 일정 수립을 위한 참고 자료로 활용해 주세요.
            """)

        if not st.session_state.has_searched:
            st.markdown('<div class="section-title">📍 추천 관광지</div>', unsafe_allow_html=True)
            with st.container(key="search_empty_notice"):
                st.info(
                    "지역, 날짜, 날씨, 혼잡도를 고른 뒤 '틈새 여행지 찾기'를 눌러 주세요.",
                    icon=":material/touch_app:"
                )

        else:
            params = st.session_state.search_params

            region_title = (
                "강원 전체" if selected_region == "전체" else selected_region
            )

            st.markdown(
                f'<div class="section-title">📍 {esc(region_title)} 추천 관광지</div>',
                unsafe_allow_html=True
            )

            # 예상 날씨 + 날씨 조건 문구
            if selected_region == "전체":
                report_regions = df_meta[region_col].dropna().unique().tolist()
            else:
                report_regions = [selected_region]

            if weather_option == "자동":
                auto_weather_values = set()

                for region in report_regions:
                    auto_weather = get_auto_weather(
                        region_name=region,
                        target_date_str=target_date_str
                    )
                    # 조회 실패 시 맑음 기본값 적용
                    auto_weather_values.add(auto_weather if auto_weather else "맑음")

                if len(auto_weather_values) == 1:
                    report_weather_label = f"{next(iter(auto_weather_values))} (자동)"
                else:
                    report_weather_label = "지역별 자동: " + ", ".join(sorted(auto_weather_values))
            else:
                report_weather_label = f"{weather_option} (직접 선택)"

            weather_html = weather_cards_html(sorted(report_regions), target_date_str)
            if weather_html:
                st.markdown(weather_html, unsafe_allow_html=True)

            # --------------------------------------
            # 1) 예측 결과 생성
            # --------------------------------------
            results = []
            region_lookup = (
                df_meta.dropna(subset=[place_col_name, region_col])
                .assign(
                    _place_key=lambda x: x[place_col_name].astype(str).str.strip()
                )
                .drop_duplicates("_place_key")
                .set_index("_place_key")[region_col]
                .to_dict()
            )

            progress_area = st.empty()

            with progress_area.container():
                progress = st.progress(0,text="혼잡도를 계산하고 있습니다...")

            total_places = len(filtered_places)

            for idx, place in enumerate(filtered_places):
                region_name = region_lookup.get(str(place).strip())
                daily_result = get_daily_peak_congestion(
                    place_name=place,
                    target_date_str=target_date_str,
                    model=model,
                    profile_lookup=profile_lookup,
                    place_monthly_weights=place_monthly_weights,
                    indoor_status_map=indoor_status_map,
                    weather_condition=weather_condition,
                    precipitation=precipitation,
                    region_name=region_name
                )

                score = daily_result["score"]
                # indoor_type = indoor_status_map.get(
                #     str(place).strip(), "야외"
                # )

                if score is None:
                    results.append({
                        "관광지명": place,
                        "혼잡 등급": daily_result["status"],
                        # "상태 색상": "⚪ 회색 (운영시간 외)",
                        "예상 최고 혼잡도": None,
                        "최고 혼잡 시간": None,
                        "유효 예측 시간": daily_result["valid_hours"],
                        "시간대별 점수": daily_result["hourly_scores"]
                    })
                else:
                    if score <= 39:
                        level = "여유"
                    elif score <= 69:
                        level = "보통"
                    elif score <= 84:
                        level = "주의"
                    else:
                        level = "혼잡"

                    results.append({
                        "관광지명": place,
                        "혼잡 등급": level,
                        # "상태 색상": badge_color,
                        "예상 최고 혼잡도": score,
                        "최고 혼잡 시간": daily_result["peak_hour"],
                        "유효 예측 시간": daily_result["valid_hours"],
                        "시간대별 점수": daily_result["hourly_scores"]
                    })

                # 진행률 표시
                if total_places:
                    progress.progress(
                        (idx + 1) / total_places,
                        text=f"혼잡도 계산 중... ({idx + 1}/{total_places})"
                    )

            progress_area.empty()

            # --------------------------------------
            # 2) 혼잡도 필터 적용
            # --------------------------------------

            for grade_name in GRADE_STYLE:
                if grade_name in congestion_filter:
                    results = [r for r in results if r["혼잡 등급"] == grade_name]
                    break

            # --------------------------------------
            # 검색 결과 요약 + 정렬 기준
            # --------------------------------------
            summary_col, sort_col = st.columns(
                [4, 1.7],
                vertical_alignment="center",
                gap="small"
            )

            with summary_col:
                st.markdown(
                    '<div class="chip-row">'
                    f'<span class="chip">📅 <b>{target_date_str}</b></span>'
                    f'<span class="chip">🌤️ 날씨 <b>{esc(report_weather_label)}</b></span>'
                    f'<span class="chip">🎯 혼잡도 <b>{esc(congestion_filter)}</b></span>'
                    f'<span class="chip">📌 <b>{len(results)}</b>곳</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with sort_col:
                with st.container(key="sort_capsule"):
                    sort_option = st.selectbox(
                        "정렬 기준",
                        [
                            "혼잡도 낮은 순",
                            "혼잡도 높은 순",
                            "별점 높은 순",
                            "리뷰 많은 순",
                            "가나다순",
                        ],
                        key="sort_option",
                        label_visibility="collapsed",
                    )



                
            # if "여유" in congestion_filter:
            #     results = [
            #         r for r in results if r["혼잡 등급"] == "여유"
            #     ]
            # elif "보통" in congestion_filter:
            #     results = [
            #         r for r in results if r["혼잡 등급"] == "보통"
            #     ]
            # elif "주의" in congestion_filter:
            #     results = [
            #         r for r in results if r["혼잡 등급"] == "주의"
            #     ]
            # elif "혼잡" in congestion_filter:
            #     results = [
            #         r for r in results if r["혼잡 등급"] == "혼잡"
            #     ]

            
            # --------------------------------------
            # 3) 정렬 옵션 및 정렬 적용
            # --------------------------------------

            # 별점 및 리뷰 수 추출
            def get_rating_review(place_name):
                place_info = place_info_lookup.get(
                    str(place_name).strip(), {}
                )

                text = str(place_info.get("평점_리뷰", ""))

                if text in ("", "정보 없음", "nan", "None"):
                    return None, None

                numbers = re.findall(
                    r"\d+(?:\.\d+)?",
                    text.replace(",", "")
                )

                rating = None
                review_count = None

                if numbers:
                    # 일반적인 평점·리뷰 표시:
                    # 예: 4.5점 · 리뷰 1,234개
                    rating = float(numbers[0])

                if len(numbers) >= 2:
                    review_count = int(float(numbers[1]))

                return rating, review_count

            # 정렬 기준 적용
            if sort_option == "혼잡도 높은 순":
                results.sort(
                    key=lambda x: (
                        x["예상 최고 혼잡도"] is not None,
                        x["예상 최고 혼잡도"]
                        if x["예상 최고 혼잡도"] is not None
                        else 0,
                    ),
                    reverse=True,
                )

            elif sort_option == "혼잡도 낮은 순":
                results.sort(
                    key=lambda x: (
                        x["예상 최고 혼잡도"] is None,
                        x["예상 최고 혼잡도"]
                        if x["예상 최고 혼잡도"] is not None
                        else 0,
                    ),
                )

            elif sort_option == "별점 높은 순":
                results.sort(
                    key=lambda x: (
                        get_rating_review(x["관광지명"])[0]
                        is not None,
                        get_rating_review(x["관광지명"])[0]
                        if get_rating_review(x["관광지명"])[0]
                        is not None
                        else 0,
                    ),
                    reverse=True,
                )

            elif sort_option == "리뷰 많은 순":
                results.sort(
                    key=lambda x: (
                        get_rating_review(x["관광지명"])[1]
                        is not None,
                        get_rating_review(x["관광지명"])[1]
                        if get_rating_review(x["관광지명"])[1]
                        is not None
                        else 0,
                    ),
                    reverse=True,
                )

            elif sort_option == "가나다순":
                results.sort(
                    key=lambda x: x["관광지명"]
                )

            st.caption(f"검색 결과 {len(results)}개")

            
            # --------------------------------------
            # 3) 카드 표시
            # --------------------------------------
            if results:
                # 페이지당 9개 관광지 표시
                page_size = 9
                total_pages = max(1, math.ceil(len(results) / page_size))

                if "result_page" not in st.session_state:
                    st.session_state.result_page = 1

                # 검색 결과가 줄어들면 현재 페이지 조정
                st.session_state.result_page = min(
                    st.session_state.result_page,
                    total_pages
                )

                start_idx = (st.session_state.result_page - 1) * page_size
                page_results = results[start_idx:start_idx + page_size]

                cols_per_row = 3

                for i in range(0, len(page_results), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j in range(cols_per_row):
                        if i + j >= len(page_results):
                            break

                        item = page_results[i + j]
                        place_name = item["관광지명"]

                        place_info = place_info_lookup.get(
                            str(place_name).strip(),
                            {
                                "운영시간": "정보 없음",
                                "주소": "정보 없음",
                                "연락처": "정보 없음",
                                "평점_리뷰": "정보 없음",
                            }
                        )

                        score = item["예상 최고 혼잡도"]
                        level = item["혼잡 등급"]
                        style = GRADE_STYLE.get(level, CLOSED_STYLE)

                        with cols[j]:
                            with st.container(key=f"card_{start_idx + i + j}"):

                                # 1. 관광지명 + 혼잡 등급
                                if score is None:
                                    grade_html = (
                                        f'<span class="grade-pill" style="background:{style["bg"]};color:{style["fg"]}">'
                                        f'{style["emoji"]} {esc(level)}</span>'
                                    )
                                    peak_html = ""
                                else:
                                    grade_html = (
                                        f'<span class="grade-pill" style="background:{style["bg"]};color:{style["fg"]}">'
                                        f'{style["emoji"]} {level}</span>'
                                        f'<span class="score-text">최고 {score:.1f}점</span>'
                                    )
                                    peak_html = (
                                        f'<p class="peak-text">가장 붐비는 시간 · {item["최고 혼잡 시간"]}시</p>'
                                    )

                                st.markdown(
                                    f'<p class="card-name">{esc(place_name)}</p>{grade_html}{peak_html}',
                                    unsafe_allow_html=True
                                )

                                # 2. 시간대별 혼잡도 그래프 (24시간)
                                if score is not None and item["시간대별 점수"]:
                                    st.plotly_chart(
                                        hourly_chart(item["시간대별 점수"], style["bar"]),
                                        use_container_width=True,
                                        config={"displayModeBar": False},
                                        theme=None,
                                        key=f"chart_{start_idx + i + j}"
                                    )

                                # 3. 운영시간 · 주소 · 연락처 · 평점

                                kakao_url = place_info.get("카카오맵_URL")
                                safe_url = ""
                                if kakao_url and str(kakao_url).strip() not in ("", "nan", "None"):
                                    safe_url = html.escape(str(kakao_url).strip(), quote=True)

                                map_link_html = (
                                    f'<a href="{safe_url}" target="_blank" '
                                    f'rel="noopener noreferrer">'
                                    '🗺️ 카카오맵에서 리뷰 확인 ↗</a>'
                                    if safe_url else ""
                                )

                                st.markdown(
                                    (
                                        '<div class="info-list">'
                                        f'<div class="info-row"><b>🕒 운영</b><span>{esc(place_info["운영시간"])}</span></div>'
                                        f'<div class="info-row"><b>📍 주소</b><span>{esc(place_info["주소"])}</span></div>'
                                        f'<div class="info-row"><b>📞 연락</b><span>{esc(place_info["연락처"])}</span></div>'
                                        f'<div class="info-row"><b>⭐ 평점</b><span>{esc(place_info["평점_리뷰"])}</span></div>'
                                        f'{map_link_html}'
                                        '</div>'
                                    ),
                                    unsafe_allow_html=True
                                )
                                                            
                                # 4. My Course 추가
                                course = st.session_state.my_course
                                if place_name in course:
                                    st.button(
                                        "코스에 담았어요",
                                        icon=":material/check:",
                                        key=f"added_{i}_{j}_{place_name}",
                                        disabled=True,
                                        use_container_width=True
                                    )
                                else:
                                    if st.button(
                                        "My Course 담기",
                                        icon=":material/add:",
                                        key=f"add_{i}_{j}_{place_name}",
                                        use_container_width=True
                                    ):
                                        if len(course) < MAX_COURSE_PLACES:
                                            add_place_dialog(
                                                place_name=place_name,
                                                visit_date=target_date_str,
                                                weather_condition=weather_condition,
                                                precipitation=precipitation,
                                                hourly_scores=item["시간대별 점수"],
                                            )
                                        else:
                                            st.warning(
                                                "코스에는 관광지를 최대 5개까지 담을 수 있어요."
                                            )

                # 페이지 번호 버튼 (카드 아래)
                st.space("small")

                current = st.session_state.result_page

                # 현재 페이지 주변의 페이지 번호 표시
                # 최대 5개 페이지 번호 + 이전/다음 버튼
                visible_pages = min(5, total_pages)

                start_page = max(
                    1,
                    min(current - 2, total_pages - visible_pages + 1)
                )
                end_page = min(total_pages, start_page + visible_pages - 1)

                # 실제로 필요한 버튼만 생성
                pagination_items = []

                if total_pages > 1:
                    pagination_items.append("prev")

                pagination_items.extend(range(start_page, end_page + 1))

                if total_pages > 1:
                    pagination_items.append("next")

                # 버튼 개수에 맞춰 컬럼 생성
                page_cols = st.columns(
                    len(pagination_items),
                    gap="small"
                )

                for col, item in zip(page_cols, pagination_items):
                    with col:
                        if item == "prev":
                            if st.button(
                                "‹",
                                key="prev_page",
                                disabled=current == 1,
                                use_container_width=True
                            ):
                                st.session_state.result_page = current - 1
                                st.rerun()

                        elif item == "next":
                            if st.button(
                                "›",
                                key="next_page",
                                disabled=current == total_pages,
                                use_container_width=True
                            ):
                                st.session_state.result_page = current + 1
                                st.rerun()

                        else:
                            if st.button(
                                str(item),
                                key=f"page_{item}",
                                type="primary" if item == current else "secondary",
                                use_container_width=True
                            ):
                                st.session_state.result_page = item
                                st.rerun()

                st.caption(
                    f"전체 {len(results)}곳 · {current}/{total_pages}페이지"
                )

            else:
                st.info(
                    "선택한 혼잡도 조건에 맞는 관광지가 없어요. 조건을 바꿔 보세요.",
                    icon=":material/search_off:"
                )

                        
            # ---------------------------------
            # 여행 추천 영상 영역
            # 페이지네이션 아래에 배치
            # ---------------------------------

            st.divider()

            # 관광지 추천 영역과 동일한 제목 흐름
            st.markdown(
                f'<div class="section-title">📍 {esc(region_title)} 여행 추천 영상</div>',
                unsafe_allow_html=True
            )

            keyword = f"{region_title} 여행"
            api_key = st.secrets.get("YOUTUBE_API_KEY", "")

            # 세션 상태 초기화
            if "youtube_videos" not in st.session_state:
                st.session_state["youtube_videos"] = None

            if "youtube_keyword" not in st.session_state:
                st.session_state["youtube_keyword"] = None

            # 지역이 변경되었거나 검색 결과가 없는 경우에만 조회
            if api_key.strip():
                if st.session_state["youtube_keyword"] != keyword:
                    try:
                        with st.spinner("여행 추천 영상을 불러오는 중..."):
                            videos = search_videos(
                                api_key.strip(),
                                keyword
                            )

                        st.session_state["youtube_videos"] = videos
                        st.session_state["youtube_keyword"] = keyword

                    except HttpError:
                        st.warning("추천 영상을 불러오지 못했습니다.")
                    except Exception as e:
                        st.error(f"영상 조회 중 오류가 발생했습니다: {e}")

                videos = st.session_state["youtube_videos"]

                if videos is not None:
                    if not videos:
                        st.info("추천할 여행 영상이 없습니다.")
                    else:
                        cols = st.columns(3)

                        for col, video in zip(cols, videos):
                            with col:
                                # 영상 제목
                                st.markdown(
                                    f"**{video['title']}**"
                                )

                                # 채널명
                                st.caption(
                                    f"채널 · {video['channel']}"
                                )

                                # 영상 플레이어
                                st.video(
                                    f"https://www.youtube.com/watch?v="
                                    f"{video['video_id']}"
                                )

                                # YouTube 원본 링크
                                st.markdown(
                                    f"[YouTube에서 보기]"
                                    f"(https://www.youtube.com/watch?v="
                                    f"{video['video_id']})"
                                )
            else:
                st.caption("추천 영상을 표시하려면 YouTube API 키를 설정해주세요.")

            st.caption(
                "추천 영상은 YouTube 검색 결과를 기반으로 제공되며, "
                "영상 재생 여부는 영상 소유자의 설정에 따라 달라질 수 있습니다."
            )


    # ==========================================
    # 오른쪽: My Course
    # ==========================================
    with course_col:
        with st.container(key="course_panel"):
            st.markdown('<div class="section-title">🧭 My Course</div>', unsafe_allow_html=True)

            course = st.session_state.my_course
            schedule = st.session_state.course_schedule

            st.caption(f"담은 관광지 {len(course)} / {MAX_COURSE_PLACES}곳")

            if course:
                course_date = next(
                    (
                        item["date"]
                        for item in schedule.values()
                        if item.get("date")
                    ),
                    None
                )

                if course_date:
                    st.caption(f"📅 여행 날짜: {course_date}")

                    if st.session_state.has_searched:
                        if str(selected_date) != str(course_date):
                            st.info(
                                f"현재 코스는 {course_date} 일정이에요. "
                                "같은 날짜의 관광지를 추가해 주세요."
                            )
                            
            st.progress(len(course) / MAX_COURSE_PLACES)

            
            st.caption("🕒 방문 일정 · 00:00–23:00")

            # 기존 일정 데이터를 컴포넌트에 전달할 형태로 변환
            timeline_items = []

            for place in st.session_state.my_course:
                item = st.session_state.course_schedule.get(place)

                if not item:
                    continue

                timeline_items.append({
                    "place": place,
                    "start": int(item["start_hour"]),
                    "duration": int(item["duration"]),
                })

            # 시간표 렌더링 및 카드 내부 삭제 버튼 이벤트 수신
            delete_event = timeline_component(
                items=timeline_items,
                key="course_timeline",
                default=None,
            )

            # 같은 클릭 이벤트가 반복 처리되지 않도록 방지
            if (
                isinstance(delete_event, dict)
                and delete_event.get("action") == "delete"
            ):
                event_id = delete_event.get("event_id")
                place = delete_event.get("place")

                last_event_id = st.session_state.get(
                    "_last_timeline_delete_event"
                )

                if event_id and event_id != last_event_id:
                    st.session_state["_last_timeline_delete_event"] = event_id

                    if place in st.session_state.my_course:
                        st.session_state.my_course.remove(place)
                        st.session_state.course_schedule.pop(place, None)
                        st.session_state.show_course_detail = False
                        st.rerun()

            #st.space("small")

            # 코스 데이터를 URL로 전달
            course_payload = []

            for place in course:
                item = schedule.get(place, {})

                course_payload.append({
                    "place": place,
                    "date": item.get("date"),
                    "start_hour": item.get("start_hour"),
                    "duration": item.get("duration"),
                    "congestion_score": item.get("congestion_score"),
                    "congestion_level": item.get("congestion_level"),
                })

            course_param = urlencode({
                "course": json.dumps(
                    course_payload,
                    ensure_ascii=False
                )
            })

            course_url = f"/courses?{course_param}"

            st.markdown(
                f"""
                <a href="{course_url}" target="_blank" rel="noopener noreferrer"
                class="course-confirm-btn">
                    🗺️ 코스 확인
                </a>
                """,
                unsafe_allow_html=True,
            )


