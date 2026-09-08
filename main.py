import argparse
from regional_risk_index import analyze_regional_risk
from vehicle_risk_clustering import train_vehicle_model
from future_risk_forecasting import run_prophet_analysis

def main():
    parser = argparse.ArgumentParser(description="교통사고 위험 분석 파이프라인")
    parser.add_argument("data", help="통합 교통사고·차량등록 CSV 경로")
    parser.add_argument("--skip-plots", action="store_true", help="대화형 차트 표시 생략")
    args = parser.parse_args()

    print("[1/3] Regional Risk Index Analysis")
    analyze_regional_risk(args.data, show_plots=not args.skip_plots)
    print("[2/3] Training RandomForest model")
    train_vehicle_model(args.data, show_plots=not args.skip_plots)
    print("[3/3] Training Prophet forecast")
    run_prophet_analysis(args.data, show_plots=not args.skip_plots)


if __name__ == "__main__":
    main()
