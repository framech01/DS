# Korea Traffic Accident Risk Analytics

##  프로젝트 개요
본 프로젝트는 교통사고 및 차량등록 데이터를 기반으로 다음의 분석 및 예측을 수행하는 데이터 과학 모델링 시스템입니다.

1. **지역별 사고 증가율 기반 위험도 분석**
2. **차량 유형 조합에 따른 지역 사고 위험도 분류**
3. **시간 흐름에 따른 사고율 예측 (시계열 기반)**

각 분석은 시각화와 함께 제공되어, 교통정책 수립 및 지역별 맞춤형 대응 전략 수립에 활용될 수 있습니다.



##  사용된 모델 및 기법

### 1. Prophet (Meta)
- **사용 목적**: 월별 교통사고율 예측
- **특징**:
  - 시계열 데이터에 특화된 모델
  - 계절성(daily seasonality) 여부에 따른 성능 비교
  - 훈련/검증 비율: 30%, 80%에 대해 실험
- **평가지표**: MAE, RMSE, R²

### 2. Random Forest Classifier
- **사용 목적**: 차량 구성비를 기반으로 지역별 사고 위험 레벨 분류
- **특징**:
  - 차량 유형 비율(승용, 승합, 화물, 특수)을 입력 특성으로 사용
  - 사고건수를 기반으로 고/중/저 위험 라벨링
  - K-Fold 교차검증을 통해 성능 안정성 확보
  - 파라미터 설정: n_estimators, n_splits 다양화 실험
- **평가지표**: Accuracy, Macro F1 Score



## 실행 방법

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py "path/to/traffic_vehicle_data.csv"
```

서버나 CI처럼 화면이 없는 환경에서는 `--skip-plots`를 추가할 수 있습니다. 입력 CSV에는 `기준월`, `시도`, `시군구`, `사고건수`, `총_계`가 필요하며 분석별 세부 차량·부상 컬럼도 사용됩니다.

## 작동 방식

### 1. 실행 파일: `main.py`
- 전체 파이프라인을 실행합니다.
  1. 지역별 사고 위험도 분석 (`regional_risk_index.py`)
  2. 차량 구성 기반 분류 모델 학습 및 평가 (`vehicle_risk_clustering.py`)
  3. 사고율 예측을 위한 Prophet 모델 훈련 및 시각화 (`future_risk_forecasting.py`)

### 2. 데이터 전처리 및 공통 설정: `common.py`
- 한글 폰트 설정
- `기준월` 컬럼에서 연도 추출 및 기본 전처리 수행

##  시각화

- 지역별 위험도 Heatmap (Basic Risk / Composite Risk)
- Prophet 예측 그래프 (Actual vs Predicted)
- Random Forest의 Feature Importance 시각화


##  참고

- Prophet: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
- Scikit-learn Random Forest: [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)

## 개선 사항

- 지역별 증감률 계산을 벡터화하고 0으로 나누는 경우를 명시적으로 처리합니다.
- 대화형 그래프를 끌 수 있어 자동화·원격 실행이 가능합니다.


