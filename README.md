# Korea Traffic Accident Risk Analytics

##  프로젝트 개요
본 프로젝트는 교통사고 및 차량등록 데이터를 기반으로 다음의 분석 및 예측을 수행하는 데이터 과학 모델링 시스템입니다.

1. **지역별 사고 증가율 기반 위험도 분석**
2. **차량 유형 조합에 따른 지역 사고 위험도 분류**
3. **시간 흐름에 따른 사고율 예측 (시계열 기반)**

각 분석은 시각화와 함께 제공되어, 교통정책 수립 및 지역별 맞춤형 대응 전략 수립에 활용될 수 있습니다.

## 사용 기술

### Language & Data Processing

- **Python 3.10+**: 데이터 검증부터 지역 위험 분석, 분류, 시계열 예측까지 전체 파이프라인을 구성합니다.
- **Pandas**: 교통사고·차량등록 CSV 로딩, `기준월` 날짜 변환, 시군구·연도별 집계, 차량 유형 구성비 계산에 사용합니다.
- **NumPy**: 안전한 비율 계산, 분위수 경계 생성, 교차검증 결과 통계와 회귀 평가 지표 계산을 담당합니다.

### Machine Learning

- **Random Forest Classifier (`scikit-learn`)**: 승용·승합·화물·특수 차량 비율을 입력으로 사고 위험을 Low/Mid/High 세 단계로 분류합니다. 비선형 관계와 특성 간 상호작용을 별도의 함수 형태 지정 없이 학습할 수 있어 사용했습니다.
- **Stratified K-Fold**: 각 fold의 위험 등급 비율을 유지하면서 설정별 Accuracy와 Macro F1을 비교합니다. Macro F1을 함께 사용해 특정 등급에 치우친 성능을 방지합니다.
- **StandardScaler**: 차량 구성비를 평균 0, 표준편차 1로 변환해 실험 입력 범위를 일관되게 관리합니다.
- **Feature Importance**: 최적 Random Forest가 어떤 차량 유형 비율을 주요 판단 근거로 사용했는지 시각화합니다.

### Time-Series Forecasting

- **Prophet**: 월별 `사고건수 / 등록차량수 × 1,000`을 목표값으로 사용해 추세 변화와 계절성을 모델링합니다.
- **시계열 순서 분할**: 데이터를 무작위로 섞지 않고 앞 구간으로 학습한 뒤 이후 구간을 예측해 실제 미래 예측 조건을 재현합니다.
- **평가 지표**: MAE로 평균 절대 오차, RMSE로 큰 오차에 대한 민감도, R²로 변동 설명력을 함께 확인합니다.

### Regional Risk Analytics

- **벡터화된 GroupBy 집계**: 시군구별 최초·최종 연도의 사고건수와 차량수를 이용해 증가율을 계산합니다.
- **Composite Risk Score**: 사고 증가율과 차량 증가율의 비율에 사망·중상·경상 심각도 가중치를 결합합니다.
- **위험 등급화**: 시도별 Risk Ratio를 Extreme, High, Moderate, Low로 구간화해 정책 우선순위를 쉽게 비교할 수 있게 합니다.

### Visualization & Runtime

- **Matplotlib / Seaborn**: 위험도 heatmap, 등급별 boxplot, 차량 구성 평균, feature importance, 실제값-예측값 그래프를 생성합니다.
- **이식 가능한 한글 폰트 설정**: Windows의 맑은 고딕과 Linux의 나눔고딕을 순서대로 탐색하며, 음수 기호 깨짐도 방지합니다.
- **CLI / Headless 실행**: `argparse`로 데이터 경로를 전달하고 `--skip-plots` 옵션으로 서버·CI 환경에서도 파이프라인을 실행할 수 있습니다.



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


