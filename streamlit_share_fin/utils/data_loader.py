
# utils/data_loader.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tomllib
from pathlib import Path


# utils 폴더의 상위 폴더 = app_v2_ey
ROOT = Path(__file__).resolve().parent.parent

# 기존 설정 파일은 app_v2_ey 폴더에 있음
with open(ROOT / "data_sources.toml", "rb") as f:
    CONFIG = tomllib.load(f)


def path(section):
    """data_sources.toml에 정의된 경로 반환"""
    return ROOT / CONFIG[section]["path"]


@st.cache_resource
def load_all_resources():
    # 1. 방문객 프로필 데이터
    df_visitor = pd.read_csv(
        path("profiles"),
        encoding="utf-8-sig"
    )
    df_visitor.columns = [
        str(col).strip() for col in df_visitor.columns
    ]

    df_visitor["profile_day"] = (
        df_visitor["profile_day"]
        .astype(str).str.strip().str.lower()
    )
    df_visitor["hour"] = pd.to_numeric(
        df_visitor["hour"], errors="coerce"
    )
    df_visitor["visitor_uv"] = pd.to_numeric(
        df_visitor["visitor_uv"], errors="coerce"
    )
    df_visitor = df_visitor.dropna(
        subset=["profile_day", "hour", "visitor_uv"]
    )

    df_visitor["_place_key"] = (
        df_visitor["place_name"].astype(str).str.strip()
    )
    df_visitor["_day_key"] = (
        df_visitor["profile_day"].astype(str).str.strip().str.lower()
    )

    profile_lookup = (
        df_visitor
        .groupby(["_place_key", "_day_key", "hour"])["visitor_uv"]
        .apply(list)
        .to_dict()
    )


    # 2. 관광지 메타데이터
    df_meta = pd.read_excel(path("spots"))

    df_meta.columns = [
        str(col).strip() for col in df_meta.columns
    ]

    # 금액 컬럼 숫자 변환
    price_cols = [
    "최소 금액 (Min/1인)",
    "최대 금액 (Max/1인)",
    ]

    for col in price_cols:
        if col in df_meta.columns:
            df_meta[col] = pd.to_numeric(
                df_meta[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("원", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
            
    # 위도·경도 컬럼 확인 및 숫자형 변환
    coordinate_cols = ["위도", "경도"]

    missing_cols = [
        col for col in coordinate_cols
        if col not in df_meta.columns
    ]

    if missing_cols:
        raise ValueError(
            f"관광지 메타데이터에 좌표 컬럼이 없습니다: {missing_cols}\n"
            f"현재 컬럼: {df_meta.columns.tolist()}"
        )

    for col in coordinate_cols:
        df_meta[col] = pd.to_numeric(
            df_meta[col], errors="coerce"
        )

    place_col = (
        "관광지명" if "관광지명" in df_meta.columns
        else (
            "관광지" if "관광지" in df_meta.columns
            else df_meta.columns[1]
        )
    )

    indoor_status_map = {}

    if "실내구분" in df_meta.columns:
        for _, row in df_meta.iterrows():
            place = str(row[place_col]).strip()
            status = str(row["실내구분"]).strip()
            indoor_status_map[place] = status


    # 3. CatBoost 모델 로드
    model_filename = path("model")

    if not model_filename.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_filename}\n"
            "model_training.py를 먼저 실행해 모델을 생성하세요."
        )

    model = joblib.load(model_filename)

    # 4. 관광지별 월간 가중치 로드
    weight_file = path("monthly_weights")
    place_monthly_weights = {}

    if os.path.exists(weight_file):
        df_weight = pd.read_excel(weight_file)
        df_weight.columns = [
            str(col).strip() for col in df_weight.columns
        ]

        for _, row in df_weight.iterrows():
            place = str(row["관광지"]).strip()
            weights = {}

            for month in range(1, 13):
                col_name = f"{month}월"

                try:
                    value = float(row[col_name])
                    weights[month] = (
                        0.0 if np.isnan(value) else value
                    )
                except (ValueError, KeyError):
                    weights[month] = 1 / 12

            place_monthly_weights[place] = weights

    return (
        df_meta,
        indoor_status_map,
        model,
        place_monthly_weights,
        profile_lookup
    )


@st.cache_data(show_spinner=False)
def build_place_info_lookup(df_meta, place_col_name):
    """관광지 상세 정보를 조회용 딕셔너리로 변환"""

    lookup = {}

    def safe_text(value):
        if pd.isna(value) or str(value).strip() in [
            "", "nan", "None"
        ]:
            return "정보 없음"
        return str(value).strip()

    for _, row in df_meta.iterrows():
        place_name = safe_text(row.get(place_col_name))

        if place_name == "정보 없음" or place_name in lookup:
            continue

        operating_hours = safe_text(
            row.get("운영시간 (휴무일 포함)")
        )
        address = safe_text(row.get("도로명주소"))
        
        phone = safe_text(row.get("전화번호"))
        if phone in ("0", "0.0"):
            phone = "정보 없음"

        rating = row.get("평점")
        review_count = row.get("리뷰수")

        rating_missing = (
            pd.isna(rating) or str(rating).strip() == ""
        )
        review_missing = (
            pd.isna(review_count)
            or str(review_count).strip() == ""
        )

        if rating_missing:
            rating_text = "정보 없음"
            stars = ""
        else:
            try:
                rating_num = float(rating)
                rating_text = f"{rating_num:.1f}"
                filled_stars = max(
                    0, min(5, round(rating_num))
                )
                stars = (
                    "★" * filled_stars
                    + "☆" * (5 - filled_stars)
                )
            except (ValueError, TypeError):
                rating_missing = True
                rating_text = "정보 없음"
                stars = ""

        if review_missing:
            review_text = "(정보 없음)"
        else:
            try:
                review_num = float(review_count)
                review_text = f"({int(review_num):,})"
            except (ValueError, TypeError):
                review_text = "(정보 없음)"

        if rating_missing and review_missing:
            rating_review = "정보 없음"
        else:
            rating_review = (
                f"{rating_text} {stars} {review_text}".strip()
            )

        kakao_url = row.get("상세URL")
        place_id = row.get("장소ID")

        if pd.notna(kakao_url) and str(kakao_url).strip():
            kakao_url = str(kakao_url).strip()
        elif pd.notna(place_id) and str(place_id).strip():
            kakao_url = (
                f"https://place.map.kakao.com/{str(place_id).strip()}"
            )
        else:
            kakao_url = None

        lookup[place_name] = {
            "운영시간": operating_hours,
            "주소": address,
            "연락처": phone,
            "평점_리뷰": rating_review,
            "카카오맵_URL": kakao_url,
        }

    return lookup
