from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_training_data(df_visitor: pd.DataFrame):
    """
    visitor profile 데이터로부터 CatBoost 학습용 X, y를 생성한다.
    visitor_uv == -1은 운영시간 외 데이터로 제외한다.
    """
    df_train_target = df_visitor.loc[
        df_visitor["visitor_uv"] != -1
    ].copy()

    if df_train_target.empty:
        raise ValueError("학습 가능한 데이터가 없습니다. visitor_uv 데이터를 확인하세요.")

    # 관광지별 방문자 수를 1~100 상대 혼잡 점수로 변환
    min_vals = df_train_target.groupby("place_name")["visitor_uv"].transform("min")
    max_vals = df_train_target.groupby("place_name")["visitor_uv"].transform("max")
    is_same = max_vals == min_vals

    # 나눗셈 오류를 피하도록 분모가 0인 경우를 별도 처리
    denominator = (max_vals - min_vals).replace(0, np.nan)
    df_train_target["congestion_score"] = (
        (df_train_target["visitor_uv"] - min_vals) / denominator
    ) * 99 + 1
    df_train_target.loc[is_same, "congestion_score"] = 50.0

    # 요일 정보 정리
    df_train_target["profile_day_lower"] = (
        df_train_target["profile_day"].astype(str).str.lower().str.strip()
    )
    weekend_days = ["saturday", "sunday"]
    df_train_target["is_weekend"] = (
        df_train_target["profile_day_lower"].isin(weekend_days).astype(int)
    )

    features = ["place_name", "profile_day", "hour", "is_weekend"]
    target = "congestion_score"

    X = df_train_target[features].copy()
    y = df_train_target[target].copy()

    # CatBoost의 범주형 변수로 사용할 값은 문자열로 통일
    X["place_name"] = X["place_name"].astype(str)
    X["profile_day"] = X["profile_day"].astype(str)

    return X, y


def train_catboost(
    df_visitor: pd.DataFrame,
    model_path="model_share/relative_congestion_score.pkl",
    test_size=0.2,
    random_state=42,
):
    """
    데이터를 랜덤 분할해 CatBoost를 학습하고 검증 성능을 계산한 뒤,
    학습된 모델을 지정 경로에 저장한다.

    Returns:
        model: 학습된 CatBoostRegressor
        metrics: MAE, RMSE, R2 검증 지표
    """
    X, y = prepare_training_data(df_visitor)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=["place_name", "profile_day"],
    )

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return model, metrics


if __name__ == "__main__":
    # 직접 실행할 때만 학습을 수행한다.
    # df_visitor를 읽는 경로는 프로젝트 환경에 맞게 지정해야 한다.
    DATA_PATH = Path("model_share/data/visitor_profiles.csv")
    MODEL_PATH = Path("model_share/relative_congestion_score.pkl")

    df_visitor = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    model, metrics = train_catboost(
        df_visitor=df_visitor,
        model_path=MODEL_PATH,
    )

    print("모델 저장 완료:", MODEL_PATH)
    print("검증 성능:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
