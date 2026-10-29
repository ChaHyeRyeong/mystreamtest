import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from utils.data_loader import load_all_resources
from pathlib import Path


# ==========================================
# 1. 페이지 설정 및 CSS 로딩
# ==========================================
st.set_page_config(
    page_title="코스 상세 보기 | 강원도 틈새여행",
    page_icon="🌿",
    layout="wide",
)

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


BASE_DIR = Path(__file__).resolve().parent.parent

for css_file in ["style.css", "course.css"]:
    css_path = BASE_DIR / css_file

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>\n{f.read()}\n</style>",
                unsafe_allow_html=True,
            )

# 상단 히어로 배너 (course.css 스타일 적용)
st.markdown(
    """
    <div class="course-hero">
        <span class="course-eyebrow">🌿 강원도 틈새여행</span>
        <div class="course-hero-title">🗺️ 코스 상세 보기</div>
        <p class="course-hero-sub">
            선택한 관광지의 위치와 구간별 이동 거리·소요 시간을 확인하세요.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. 메인 화면에서 전달한 코스 정보 읽기
# ==========================================
course_param = st.query_params.get("course", "")

if isinstance(course_param, list):
    course_param = course_param[0] if course_param else ""

if not course_param:
    st.warning("전달된 코스 정보가 없습니다.")
    st.info("메인 화면에서 관광지를 코스에 추가한 후 '코스 확인'을 눌러주세요.")
    st.stop()

try:
    course = json.loads(course_param)
except (json.JSONDecodeError, TypeError):
    st.error("코스 정보를 읽을 수 없습니다. 메인 화면에서 다시 코스를 열어주세요.")
    st.stop()

if not isinstance(course, list) or not course:
    st.warning("코스에 등록된 관광지가 없습니다.")
    st.stop()

course = sorted(
    course,
    key=lambda item: (
        item.get("date") or "",
        item.get("start_hour")
        if item.get("start_hour") is not None
        else 24
    )
)

# 최대 5곳까지만 처리
course = course[:5]


# ==========================================
# 3. 관광지 메타데이터 및 좌표 불러오기
# ==========================================
df_meta, indoor_status_map, model, place_monthly_weights, profile_lookup = (
    load_all_resources()
)

df_meta.columns = [str(col).strip() for col in df_meta.columns]

if "관광지명" in df_meta.columns:
    place_col = "관광지명"
elif "관광지" in df_meta.columns:
    place_col = "관광지"
else:
    st.error("관광지 메타데이터에서 관광지명 컬럼을 찾을 수 없습니다.")
    st.stop()

required_cols = [place_col, "위도", "경도"]
missing_cols = [col for col in required_cols if col not in df_meta.columns]

if missing_cols:
    st.error(
        f"관광지 메타데이터에 필요한 컬럼이 없습니다: {missing_cols}"
    )
    st.stop()

df_meta["위도"] = pd.to_numeric(df_meta["위도"], errors="coerce")
df_meta["경도"] = pd.to_numeric(df_meta["경도"], errors="coerce")
df_meta[place_col] = df_meta[place_col].astype(str).str.strip()

price_lookup = {}
price_cols = ["최소 금액 (Min/1인)", "최대 금액 (Max/1인)"]

if all(col in df_meta.columns for col in price_cols):
    price_lookup = (
        df_meta.drop_duplicates(subset=[place_col])
        .set_index(place_col)[price_cols]
        .to_dict("index")
    )

coordinate_lookup = (
    df_meta.dropna(subset=["위도", "경도"])
    .drop_duplicates(subset=[place_col])
    .set_index(place_col)[["위도", "경도"]]
    .to_dict("index")
)

route_places = []

for item in course:
    if not isinstance(item, dict):
        continue

    place_name = str(item.get("place", "")).strip()
    if not place_name:
        continue

    coordinate = coordinate_lookup.get(place_name)

    route_places.append({
        "place": place_name,
        "date": item.get("date"),
        "start_hour": item.get("start_hour"),
        "duration": item.get("duration"),
        "congestion_score": item.get("congestion_score"),
        "congestion_level": item.get("congestion_level"),
        "lat": coordinate["위도"] if coordinate else None,
        "lon": coordinate["경도"] if coordinate else None,
    })

if not route_places:
    st.warning("표시할 관광지가 없습니다.")
    st.stop()

missing_places = [
    item["place"]
    for item in route_places
    if item["lat"] is None or item["lon"] is None
]

if missing_places:
    st.error(
        "다음 관광지의 좌표를 찾지 못했습니다. "
        "관광지명과 메타데이터의 관광지명이 일치하는지 확인해주세요.\n\n"
        + ", ".join(missing_places)
    )
    st.stop()

# ==========================================
# 4. 코스 일정 표시 (순수 HTML 카드 렌더링)
# ==========================================
st.markdown('<div class="course-section-title">📍 여행 코스 일정</div>', unsafe_allow_html=True)

for idx, item in enumerate(route_places, start=1):
    date_text = item.get("date") or "날짜 미지정"
    start_hour = item.get("start_hour")
    duration = item.get("duration")

    if start_hour is not None and duration is not None:
        schedule_text = f"{int(start_hour):02d}:00 방문 · {duration}시간 체류"
    else:
        schedule_text = "방문 시간 미지정"

    # 시작시간 기준 혼잡도 표시
    congestion_score = item.get("congestion_score")
    congestion_level = item.get("congestion_level")

    congestion_colors = {
        "여유": ("#D8F1E3", "#2F7A55", "🌿"),
        "보통": ("#FFF3C4", "#8A6A12", "🌼"),
        "주의": ("#FFE0C7", "#A2541C", "🍊"),
        "혼잡": ("#FBD5D5", "#A33A3A", "🔥"),
    }

    if congestion_score is not None and congestion_level in congestion_colors:
        bg, fg, emoji = congestion_colors[congestion_level]

        congestion_html = (
            f'<div style="display:inline-block;'
            f'background:{bg};color:{fg};'
            f'padding:5px 10px;border-radius:8px;'
            f'font-size:13px;font-weight:700;'
            f'margin-top:6px;">'
            f'🕒 방문시간 혼잡도 · {emoji} {congestion_level} '
            f'{float(congestion_score):.1f}점'
            f'</div>'
        )
    else:
        congestion_html = (
            '<div style="font-size:13px;color:#7A8A83;'
            'margin-top:6px;">'
            '🕒 방문시간 혼잡도 · 예측 정보 없음'
            '</div>'
        )

    # 비용 정보 조회
    price_info = price_lookup.get(item["place"], {})
    min_price = price_info.get("최소 금액 (Min/1인)")
    max_price = price_info.get("최대 금액 (Max/1인)")
    price_html = ""

    if pd.notna(min_price) and pd.notna(max_price):
        if min_price == 0 and max_price == 0:
            price_text = "0원"
        else:
            price_text = f"{int(min_price):,}~{int(max_price):,}원/인"

        price_html = (
            f'<div class="course-price-badge">'
            f'💵 예상 비용: {price_text}'
            f'</div>'
        )
    elif pd.notna(min_price):
        price_text = f"최소 {int(min_price):,}원/인"
        price_html = f'<div class="course-price-badge">💵 예상 비용: {price_text}</div>'
    elif pd.notna(max_price):
        price_text = f"최대 {int(max_price):,}원/인"
        price_html = f'<div class="course-price-badge">💵 예상 비용: {price_text}</div>'

    # 순수 HTML 코드로 카드를 감싸서 Streamlit 기본 컨테이너 간섭 제거
    st.markdown(
        f"""
        <div class="pure-course-card">
            <div class="course-item-flex">
                <div class="course-item-left">
                    <span class="course-place-number">{idx}</span>
                    <span class="course-place-name">{item['place']}</span>
                </div>
                <div class="course-item-right">
                    <div class="course-meta-text">📅 {date_text} &nbsp;|&nbsp; 🕒 {schedule_text}</div>
                    {congestion_html}
                    {price_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ==========================================
# 5. TMap 경로 조회 함수
# ==========================================
TMAP_API_KEY = st.secrets.get("TMAP_API_KEY", "")
KAKAO_MAP_API_KEY = st.secrets.get("KAKAO_MAP_API_KEY", "")


def format_distance(meters):
    if meters is None:
        return "-"
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{meters:.0f} m"


def format_duration(seconds):
    if seconds is None:
        return "-"
    minutes = round(seconds / 60)
    hours, mins = divmod(minutes, 60)

    if hours:
        return f"{hours}시간 {mins}분"
    return f"{mins}분"


def get_visit_datetime(item):
    visit_date = item.get("date")
    start_hour = item.get("start_hour")

    if visit_date is None or start_hour is None:
        return None

    try:
        visit_date = datetime.strptime(str(visit_date)[:10], "%Y-%m-%d")
        return visit_date + timedelta(hours=float(start_hour))
    except (ValueError, TypeError):
        return None


def check_schedule_conflict(current_item, next_item, travel_time_sec):
    current_visit = get_visit_datetime(current_item)
    next_visit = get_visit_datetime(next_item)
    duration = current_item.get("duration")

    if (
        current_visit is None
        or next_visit is None
        or duration is None
        or travel_time_sec is None
    ):
        return None

    try:
        departure_time = current_visit + timedelta(hours=float(duration))
        arrival_time = departure_time + timedelta(seconds=float(travel_time_sec))
    except (ValueError, TypeError):
        return None

    if arrival_time > next_visit:
        return (
            f"⚠️ {next_item['place']}의 방문시간을 다시 확인해주세요! "
            f"예상 도착시간은 {arrival_time:%H:%M}, "
            f"방문 예정시간은 {next_visit:%H:%M}입니다."
        )

    return None


def get_driving_route(start, end):
    url = "https://apis.openapi.sk.com/tmap/routes?version=1"
    headers = {
        "appKey": TMAP_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "startX": str(start["lon"]),
        "startY": str(start["lat"]),
        "endX": str(end["lon"]),
        "endY": str(end["lat"]),
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
        "searchOption": "0",
        "startName": start["place"],
        "endName": end["place"],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])

    if not features:
        raise ValueError("TMap에서 경로 결과를 반환하지 않았습니다.")

    total_distance = None
    total_time = None
    path = []

    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if total_distance is None and properties.get("totalDistance") is not None:
            total_distance = properties["totalDistance"]

        if total_time is None and properties.get("totalTime") is not None:
            total_time = properties["totalTime"]

        if geometry_type == "LineString":
            path.extend([
                [coord[1], coord[0]]
                for coord in coordinates
                if len(coord) >= 2
            ])

    if total_distance is None or total_time is None:
        raise ValueError("경로의 거리 또는 소요 시간 정보가 없습니다.")

    return {
        "distance": float(total_distance),
        "duration": float(total_time),
        "path": path,
    }


# ==========================================
# 6. 구간별 경로 계산
# ==========================================
if len(route_places) < 2:
    st.info("이동 경로를 확인하려면 관광지를 2곳 이상 추가해주세요.")
    route_results = []
else:
    if not TMAP_API_KEY:
        st.error(
            "TMAP_API_KEY가 설정되어 있지 않습니다. "
            "Streamlit secrets 설정을 확인해주세요."
        )
        st.stop()

    route_results = []

    with st.spinner("관광지 간 이동 경로를 조회하고 있습니다..."):
        for idx in range(len(route_places) - 1):
            start = route_places[idx]
            end = route_places[idx + 1]

            try:
                result = get_driving_route(start, end)
                schedule_warning = check_schedule_conflict(
                    current_item=start,
                    next_item=end,
                    travel_time_sec=result["duration"],
                )

                route_results.append({
                    "start": start["place"],
                    "end": end["place"],
                    **result,
                    "error": None,
                    "schedule_warning": schedule_warning,
                })

            except requests.RequestException as exc:
                route_results.append({
                    "start": start["place"],
                    "end": end["place"],
                    "distance": None,
                    "duration": None,
                    "path": [],
                    "error": f"경로 API 요청 실패: {exc}",
                })

            except (ValueError, KeyError, TypeError) as exc:
                route_results.append({
                    "start": start["place"],
                    "end": end["place"],
                    "distance": None,
                    "duration": None,
                    "path": [],
                    "error": str(exc),
                })


# ==========================================
# 7. 전체 이동 거리·시간 요약 및 구간별 이동 정보
# ==========================================

# 코스 전체 예상 비용 합계
total_min_price = 0
total_max_price = 0

for item in course:
    price_info = price_lookup.get(item["place"], {})

    min_price = price_info.get("최소 금액 (Min/1인)")
    max_price = price_info.get("최대 금액 (Max/1인)")

    if pd.notna(min_price):
        total_min_price += min_price

    if pd.notna(max_price):
        total_max_price += max_price

# 0~0원이면 0원으로 표시
if total_min_price == 0 and total_max_price == 0:
    total_price_text = "0원"
else:
    total_price_text = (
        f"{int(total_min_price):,}~{int(total_max_price):,}원"
    )

    
valid_results = [
    result for result in route_results
    if result["distance"] is not None and result["duration"] is not None
]

if valid_results:
    total_distance = sum(r["distance"] for r in valid_results)
    total_duration = sum(r["duration"] for r in valid_results)

    with st.container(key="route_summary_box"):
        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric("이동 구간", f"{len(valid_results)}개")

        with metric2:
            st.metric("총 이동 거리", format_distance(total_distance))

        with metric3:
            st.metric("총 이동 시간", format_duration(total_duration))

        with metric4:
            st.metric("예상 총 비용", total_price_text)

    # st.caption(
    #     "총 이동 시간은 각 구간의 TMap 소요 시간을 합산한 값입니다. "
    #     "관광지 체류 시간은 포함하지 않습니다."
    # )
    st.markdown(
        """
        <small style="color: #7A8A83;">
        총 이동 시간은 각 구간의 TMap 소요 시간을 합산한 값입니다. 관광지 체류 시간은 포함하지 않습니다.<br>
        방문시간 혼잡도와 예상 비용은 예상치입니다. 정확하지 않을 수 있습니다.<br>
        </small>
        """,
        unsafe_allow_html=True
    )
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="course-section-title">🚗 구간별 이동 정보</div>', unsafe_allow_html=True)

if not route_results:
    st.caption("계산할 이동 구간이 없습니다.")
else:
    for idx, result in enumerate(route_results, start=1):
        with st.container(key=f"route_segment_{idx}"):
            st.markdown(
                f'<div class="route-segment-title">'
                f'구간 {idx}. {result["start"]} '
                f'<span class="route-arrow">→</span> '
                f'{result["end"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if result["error"]:
                st.warning(result["error"])
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    departure_text = "시간 미지정"
                    current_item = route_places[idx - 1]

                    visit_datetime = get_visit_datetime(current_item)
                    duration = current_item.get("duration")

                    if visit_datetime is not None and duration is not None:
                        departure_datetime = visit_datetime + timedelta(
                            hours=float(duration)
                        )
                        departure_text = departure_datetime.strftime("%H:%M")

                    st.metric("출발 예정", departure_text)

                with col2:
                    st.metric("이동 거리", format_distance(result["distance"]))

                with col3:
                    st.metric("예상 소요 시간", format_duration(result["duration"]))

                if result.get("schedule_warning"):
                    st.warning(result["schedule_warning"])


# ==========================================
# 8. Kakao Map 표시
# ==========================================
#st.divider()
st.markdown('<div class="course-section-title">🗺️ 코스 지도</div>', unsafe_allow_html=True)

if not KAKAO_MAP_API_KEY:
    st.warning(
        "KAKAO_MAP_API_KEY가 설정되어 있지 않아 지도를 표시할 수 없습니다. "
        "Streamlit secrets 설정을 확인해주세요."
    )
else:
    map_places = [
        {
            "name": item["place"],
            "lat": item["lat"],
            "lon": item["lon"],
            "order": idx,
        }
        for idx, item in enumerate(route_places, start=1)
    ]

    map_routes = [
        {
            "path": result["path"],
            "start": result["start"],
            "end": result["end"],
        }
        for result in valid_results
        if result["path"]
    ]

    places_json = json.dumps(map_places, ensure_ascii=False)
    routes_json = json.dumps(map_routes, ensure_ascii=False)
    kakao_key_json = json.dumps(KAKAO_MAP_API_KEY)

    map_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                font-family: 'Noto Sans KR', sans-serif;
            }}
            #map {{
                width: 100%;
                height: 620px;
                border-radius: 16px;
                border: 1px solid #DCEBE2;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <script>
            const places = {places_json};
            const routes = {routes_json};
            const kakaoKey = {kakao_key_json};

            const script = document.createElement("script");
            script.src =
                "https://dapi.kakao.com/v2/maps/sdk.js?appkey="
                + encodeURIComponent(kakaoKey)
                + "&autoload=false";

            script.onload = function () {{
                kakao.maps.load(function () {{
                    const container = document.getElementById("map");

                    const center = places.length
                        ? new kakao.maps.LatLng(
                            places[0].lat,
                            places[0].lon
                          )
                        : new kakao.maps.LatLng(37.7, 128.2);

                    const map = new kakao.maps.Map(container, {{
                        center: center,
                        level: 9
                    }});

                    const bounds = new kakao.maps.LatLngBounds();

                    places.forEach(function (place) {{
                        const position = new kakao.maps.LatLng(
                            place.lat,
                            place.lon
                        );

                        const marker = new kakao.maps.Marker({{
                            map: map,
                            position: position
                        }});

                        const overlay = new kakao.maps.CustomOverlay({{
                            position: position,
                            yAnchor: 2.2,
                            content:
                                '<div style="padding:6px 10px;'
                                + 'background:#ffffff;'
                                + 'border:1px solid #3F8A6E;'
                                + 'border-radius:8px;'
                                + 'font-size:12px;'
                                + 'font-weight:700;'
                                + 'color:#2E5E4E;'
                                + 'box-shadow: 0 4px 12px rgba(63, 138, 110, 0.15);'
                                + 'white-space:nowrap;">'
                                + place.order + '. '
                                + place.name
                                + '</div>'
                        }});

                        overlay.setMap(map);
                        bounds.extend(position);
                    }});

                    routes.forEach(function (route) {{
                        if (!route.path || route.path.length < 2) {{
                            return;
                        }}

                        const linePath = route.path.map(function (point) {{
                            return new kakao.maps.LatLng(
                                point[0],
                                point[1]
                            );
                        }});

                        const polyline = new kakao.maps.Polyline({{
                            path: linePath,
                            strokeWeight: 5,
                            strokeColor: "#3F8A6E",
                            strokeOpacity: 0.85,
                            strokeStyle: "solid"
                        }});

                        polyline.setMap(map);

                        linePath.forEach(function (point) {{
                            bounds.extend(point);
                        }});
                    }});

                    if (places.length > 1) {{
                        map.setBounds(bounds);
                    }}
                }});
            }};

            script.onerror = function () {{
                document.getElementById("map").innerHTML =
                    "카카오 지도 SDK 로딩 실패<br>"
                    + "요청 주소: " + script.src;
                console.error("Kakao Maps SDK loading failed:", script.src);
            }};

            document.head.appendChild(script);
        </script>
    </body>
    </html>
    """

    components.html(map_html, height=640, scrolling=False)


# ==========================================
# 9. 안내
# ==========================================
st.markdown(
    '<div class="course-map-caption">※ 이동 거리는 자동차 경로 기준입니다. 실제 이동 시간은 교통 상황과 출발 시점에 따라 달라질 수 있습니다.</div>',
    unsafe_allow_html=True,
)