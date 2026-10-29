
from utils.predict_congestion import (
    load_weather_data,
    get_auto_weather,
    get_weather_multiplier_detailed,
)

import pandas as pd

# 1. CSV 로드
df = load_weather_data()

print("=" * 60)
print("1. CSV 데이터 확인")
print("행 수:", len(df))
print("날씨별 건수:")
print(df["예측"].value_counts(dropna=False))

# 2. 지역·월일별로 서로 다른 날씨가 몇 종류인지 확인
valid_weather = ["맑음", "비", "눈", "폭염", "한파"]

valid = df[df["예측"].isin(valid_weather)].copy()

groups = (
    valid.groupby(["지역_키", "월일_키"])["예측"]
    .agg(lambda x: sorted(set(x)))
    .reset_index(name="날씨목록")
)

# 서로 다른 날씨가 하나로 확정되는 사례만 테스트
groups["날씨종류수"] = groups["날씨목록"].str.len()
cases = groups[groups["날씨종류수"] == 1]

print("\n날씨가 하나로 확정되는 지역·날짜 조합:", len(cases))

# 3. 맑음·비·눈·폭염·한파별 매칭 사례 확인
print("\n" + "=" * 60)
print("2. 자동 날씨 매칭 테스트")

for weather in valid_weather:
    candidates = cases[
        cases["날씨목록"].apply(lambda x: x[0] == weather)
    ]

    if candidates.empty:
        print(f"[확인 필요] {weather}: 테스트 사례 없음")
        continue

    row = candidates.iloc[0]
    region = row["지역_키"]
    month_day = row["월일_키"]

    # 윤년 날짜도 처리할 수 있도록 2024년 사용
    target_date = f"2024-{month_day}"

    actual = get_auto_weather(
        region_name=region,
        target_date_str=target_date,
    )

    expected = weather
    result = "PASS" if actual == expected else "FAIL"

    print(
        f"[{result}] 지역={region}, 날짜={target_date}, "
        f"기대={expected}, 결과={actual}"
    )

# 4. 지역명 뒤에 '시'가 붙어도 매칭되는지 확인
print("\n" + "=" * 60)
print("3. 지역명 접미사 테스트")

if not cases.empty:
    row = cases.iloc[0]
    region = row["지역_키"]
    target_date = f"2024-{row['월일_키']}"

    normal = get_auto_weather(region, target_date)
    alias = get_auto_weather(region + "시", target_date)

    print("원래 지역명:", region, "=>", normal)
    print("접미사 추가:", region + "시", "=>", alias)
    print("결과:", "PASS" if normal == alias else "FAIL")

# 5. 날씨 보정 배율 확인
print("\n" + "=" * 60)
print("4. 날씨별 실내·야외 보정 배율")

for weather in ["맑음", "비", "눈", "폭염", "한파"]:
    outdoor = get_weather_multiplier_detailed(
        "야외", weather, 0.0
    )
    indoor = get_weather_multiplier_detailed(
        "실내", weather, 0.0
    )

    print(
        f"{weather}: 야외 ×{outdoor:.2f}, "
        f"실내 ×{indoor:.2f}"
    )
