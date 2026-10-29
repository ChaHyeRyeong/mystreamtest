
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import tomllib

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_weather_data():
    """날씨 CSV를 한 번만 읽어서 재사용한다."""
    config = tomllib.loads(
        (ROOT / "data_sources.toml").read_text(encoding="utf-8")
    )

    csv_path = ROOT / config["weather"]["path"]

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df["지역_키"] = (
        df["지역"].fillna("").astype(str).str.strip().str[:2]
    )
    df["월일_키"] = (
        df["예측월일"].fillna("").astype(str).str.strip()
    )
    df["예측"] = (
        df["예측"].fillna("").astype(str).str.strip()
    )

    return df


def get_auto_weather(region_name, target_date_str):
    """지역과 월일을 기준으로 CSV의 날씨 예측값을 조회한다."""
    if region_name is None or pd.isna(region_name):
        return None

    region_key = str(region_name).strip()[:2]

    if not region_key:
        return None

    target_month_day = datetime.strptime(
        target_date_str, "%Y-%m-%d"
    ).strftime("%m-%d")

    df = load_weather_data()

    matched = df.loc[
        (df["지역_키"] == region_key)
        & (df["월일_키"] == target_month_day),
        "예측"
    ]

    valid = matched[matched.isin(["맑음", "비", "눈", "폭염", "한파"])]

    if valid.empty:
        return None

    # 동일 지역·월일에 서로 다른 예측값이 있는지 확인
    unique_weather = valid.unique()

    if len(unique_weather) > 1:
        # 임의로 첫 번째 값을 선택하지 않고 매칭 실패로 처리
        return None

    return unique_weather[0]


def get_weather_multiplier_detailed(
    is_indoor,
    weather_condition,
    precipitation
):
    indoor_flag = "실내" in str(is_indoor)

    if (
        precipitation > 0
        or weather_condition in ["비", "눈", "비/눈"]
    ):
        return 1.20 if indoor_flag else 0.75

    elif weather_condition in ["폭염", "한파"]:
        return 1.10 if indoor_flag else 0.85

    else:
        return 0.90 if indoor_flag else 1.00


def get_congestion_level(score):
    if score <= 39:
        return {
            "level": "여유",
            "color": "녹색",
            "color_code": "green"
        }
    elif score <= 69:
        return {
            "level": "보통",
            "color": "노랑",
            "color_code": "yellow"
        }
    elif score <= 84:
        return {
            "level": "주의",
            "color": "주황",
            "color_code": "orange"
        }
    else:
        return {
            "level": "혼잡",
            "color": "빨강",
            "color_code": "red"
        }


def predict_congestion_final(
    place_name,
    target_date_str,
    hour,
    model,
    profile_lookup,
    place_monthly_weights,
    indoor_status_map,
    weather_condition="맑음",
    precipitation=0.0,
    region_name=None
):
    dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    target_month = dt.month
    profile_day = dt.strftime("%A").lower()

    is_weekend = int(
        profile_day in ["saturday", "sunday"]
    )
    place_clean = str(place_name).strip()

    profile_values = profile_lookup.get(
        (place_clean, profile_day, hour)
    )

    if not profile_values:
        return {
            "place_name": place_name,
            "date": target_date_str,
            "hour": hour,
            "status": "프로필 데이터 없음",
            "final_score": None
        }

    if all(value == -1 for value in profile_values):
        return {
            "place_name": place_name,
            "date": target_date_str,
            "hour": hour,
            "status": "운영 시간 외",
            "final_score": None
        }

    input_data = pd.DataFrame({
        "place_name": [place_name],
        "profile_day": [profile_day],
        "hour": [hour],
        "is_weekend": [is_weekend]
    })

    base_score = model.predict(input_data)[0]

    if place_clean not in place_monthly_weights:
        month_multiplier = 1.0
        match_status = "월별데이터 미매칭 (기본 1.0)"
    else:
        l1_ratio = place_monthly_weights[
            place_clean
        ].get(target_month, 1 / 12)

        month_multiplier = max(
            0.3, min(2.0, l1_ratio * 12)
        )
        match_status = "정상 매칭"

    indoor_type = indoor_status_map.get(
        place_clean, "야외"
    )

    # 자동 선택일 때만 CSV 날씨를 조회한다.
    if weather_condition == "자동":
        auto_weather = get_auto_weather(
            region_name=region_name,
            target_date_str=target_date_str
        )

        if auto_weather is not None:
            weather_condition = auto_weather
        else:
            # 매칭 실패 시 기존 기본 날씨 적용
            weather_condition = "맑음"

    weather_multiplier = get_weather_multiplier_detailed(
        indoor_type,
        weather_condition,
        precipitation
    )


    final_score = (
        base_score
        * month_multiplier
        * weather_multiplier
    )
    final_score = max(1.0, min(100.0, final_score))

    congestion_info = get_congestion_level(final_score)

    return {
        "place_name": place_name,
        "place_type": indoor_type,
        "date": target_date_str,
        "hour": hour,
        "weather": (
            f"{weather_condition} (강수: {precipitation}mm)"
        ),
        "match_status": match_status,
        "base_score": round(base_score, 2),
        "month_multiplier": round(month_multiplier, 2),
        "weather_multiplier": weather_multiplier,
        "final_score": round(final_score, 2),
        "congestion_level": congestion_info["level"],
        "congestion_color": congestion_info["color"],
        "color_code": congestion_info["color_code"]
        # "weather_match_status": weather_match_status
    }


def get_daily_peak_congestion(
    place_name,
    target_date_str,
    model,
    profile_lookup,
    place_monthly_weights,
    indoor_status_map,
    weather_condition="맑음",
    precipitation=0.0,
    region_name=None
):

    dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    target_month = dt.month
    profile_day = dt.strftime("%A").lower()

    is_weekend = int(
        profile_day in ["saturday", "sunday"]
    )
    place_clean = str(place_name).strip()

    valid_hours = []

    for hour in range(24):
        profile_values = profile_lookup.get(
            (place_clean, profile_day, hour)
        )

        if not profile_values:
            continue

        if all(value == -1 for value in profile_values):
            continue

        valid_hours.append(hour)

    if not valid_hours:
        return {
            "score": None,
            "peak_hour": None,
            "valid_hours": 0,
            "hourly_scores": [],
            "status": "운영시간 외"
        }

    input_data = pd.DataFrame({
        "place_name": [place_name] * len(valid_hours),
        "profile_day": [profile_day] * len(valid_hours),
        "hour": valid_hours,
        "is_weekend": [is_weekend] * len(valid_hours)
    })

    base_scores = model.predict(input_data)

    if place_clean not in place_monthly_weights:
        month_multiplier = 1.0
    else:
        l1_ratio = place_monthly_weights[
            place_clean
        ].get(target_month, 1 / 12)

        month_multiplier = max(
            0.3, min(2.0, l1_ratio * 12)
        )

    indoor_type = indoor_status_map.get(
        place_clean, "야외"
    )

    # 날씨 매칭 상태 초기화
    weather_match_status = "수동 선택"

    # 자동 날씨 선택 시 CSV 조회
    if weather_condition == "자동":
        auto_weather = get_auto_weather(
            region_name=region_name,
            target_date_str=target_date_str
        )

        if auto_weather is not None:
            weather_condition = auto_weather
            weather_match_status = "CSV 매칭 성공"
        else:
            weather_condition = "맑음"
            weather_match_status = "CSV 매칭 실패 - 맑음 기본값"

    # 날씨 보정 배율 계산
    weather_multiplier = get_weather_multiplier_detailed(
        indoor_type,
        weather_condition,
        precipitation
    )

    # print("DEBUG weather_match_status:", weather_match_status)
    
    hourly_results = []

    for hour, base_score in zip(valid_hours, base_scores):
        final_score = (
            float(base_score)
            * month_multiplier
            * weather_multiplier
        )
        final_score = max(1.0, min(100.0, final_score))

        hourly_results.append({
            "hour": hour,
            "final_score": round(final_score, 2)
        })

    peak_result = max(
        hourly_results,
        key=lambda x: x["final_score"]
    )

    # print(
    #     f"[혼잡도 검증] 관광지={place_clean}, "
    #     f"날씨={weather_condition}, "
    #     f"월별배율={month_multiplier:.3f}, "
    #     f"날씨배율={weather_multiplier:.3f}, "
    #     f"최종점수={final_score:.2f}"
    # )

    return {
        "score": peak_result["final_score"],
        "peak_hour": peak_result["hour"],
        "valid_hours": len(hourly_results),
        "hourly_scores": hourly_results,
        "status": "정상",
        "weather_match_status": weather_match_status
    }
