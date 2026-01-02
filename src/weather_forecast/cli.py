import argparse
from pathlib import Path

#import from all previous files
from valais_weather.download import download_data
from valais_weather.clean import clean_raw_data
from valais_weather.interpolate import run_interpolation_pipeline
from valais_weather.merge import merge_interpolated_variables
from valais_weather.train_classifier import train_classifier_models
from valais_weather.train_regressor import train_regressors
from valais_weather.forecast import run_forecast

# ── DEFAULT PATHS ─────────────────────────────────────────────
DATA_DIR = Path("data")

DEFAULT_RAW_DIR = DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_INTERPOLATED_DIR = DATA_DIR / "interpolated"
DEFAULT_CLEAN_PARQUET = DATA_DIR / "clean/valais_clean.parquet"

DEFAULT_CLASSIFIER_DIR = DATA_DIR / "models/classifier"
DEFAULT_REGRESSOR_DIR = DATA_DIR / "models/regressor"


# ── ARGUMENT PARSER ───────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(
        description="Valais Weather ML Pipeline"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── DOWNLOAD ───────────────────────────
    p = sub.add_parser("download")
    p.add_argument("--stations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--overwrite", action="store_true")

    # ── CLEAN ──────────────────────────────
    p = sub.add_parser("clean")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)

    # ── INTERPOLATE ────────────────────────
    p = sub.add_parser("interpolate")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_INTERPOLATED_DIR)
    p.add_argument(
        "--variogram",
        choices=["fixed", "fitted"],
        default="fitted",
    )

    # ── MERGE ──────────────────────────────
    p = sub.add_parser("merge")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INTERPOLATED_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_CLEAN_PARQUET)

    # ── TRAIN CLASSIFIER ───────────────────
    p = sub.add_parser("train-classifier")
    p.add_argument("--weather-parquet", type=Path, default=DEFAULT_CLEAN_PARQUET)
    p.add_argument("--stations", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    p.add_argument("--seed", type=int, default=42)

    # ── TRAIN REGRESSOR ────────────────────
    p = sub.add_parser("train-regressor")
    p.add_argument("--weather-parquet", type=Path, default=DEFAULT_CLEAN_PARQUET)
    p.add_argument("--stations", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_REGRESSOR_DIR)
    p.add_argument("--seed", type=int, default=42)

    # ── FORECAST ───────────────────────────
    p = sub.add_parser("forecast")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--weather-parquet", type=Path, default=DEFAULT_CLEAN_PARQUET)
    p.add_argument("--stations", type=Path, required=True)
    p.add_argument("--classifier", type=Path, required=True)
    p.add_argument("--regressor", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)

    return parser


# ── MAIN ─────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    if args.command == "download":
        download_data(
            stations_file=args.stations,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

    elif args.command == "clean":
        clean_raw_data(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )

    elif args.command == "interpolate":
        run_interpolation_pipeline(
            input_dir=str(args.input_dir),
            output_dir=str(args.output_dir),
            variables=None,
            variogram_mode=args.variogram,
        )

    elif args.command == "merge":
        merge_interpolated_variables(
            input_dir=str(args.input_dir),
            output_path=str(args.output),
        )

    elif args.command == "train-classifier":
        train_classifier_models(
            parquet_path=str(args.weather_parquet),
            stations_csv=str(args.stations),
            model_dir=str(args.model_dir),
            architectures=None,  
            seed=args.seed,
        )

    elif args.command == "train-regressor":
        train_regressors(
            weather_parquet=str(args.weather_parquet),
            stations_csv=str(args.stations),
            model_dir=str(args.model_dir),
            architectures=None, 
            seed=args.seed,
        )

    elif args.command == "forecast":
        table = run_forecast(
            horizon=args.horizon,
            weather_parquet=str(args.weather_parquet),
            stations_csv=str(args.stations),
            classifier_path=str(args.classifier),
            regressor_path=str(args.regressor),
            scaler_path=str(args.scaler),
            seed=args.seed,
        )
        print(table)


if __name__ == "__main__":
    main()
