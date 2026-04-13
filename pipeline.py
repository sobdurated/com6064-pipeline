import argparse
import json
import os
import runpy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pymongo import MongoClient


PIPELINE_STEPS: List[Tuple[str, str]] = [
    ("fetch", "fetch.py"), #yusuf's task
    ("preprocess", "preprocess.py"), #hamza's task
    ("sentiment", "sentiment_analysis.py"), #ammaar's task
    ("geospatial", "geospatial.py"), #adel's task
    ("sentiment_aggregation", "sentiment_aggregation.py") #ebraheem's task
]

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "COM6064")


def _run_single_step(step_name: str, script_name: str, input_data: Any, context: Dict[str, Any], base_dir: Path) -> Any:
    script_path = base_dir / script_name

    if not script_path.exists():
        print(f"step skipped: {step_name} | missing file: {script_name}")
        return input_data

    print(f"running step: {step_name} | script: {script_name}")

    namespace = runpy.run_path(
        str(script_path),
        init_globals={
            "INPUT_DATA": input_data,
            "CONTEXT": context,
            "STEP_NAME": step_name,
        },
    )

    run_fn = namespace.get("run")
    if callable(run_fn):
        return run_fn(input_data, context)

    if "OUTPUT_DATA" in namespace:
        return namespace["OUTPUT_DATA"]

    print(f"step finished with no output contract: {step_name} | using passthrough data")
    return input_data


def _parse_initial_input(raw_input: str | None) -> Any:
    if raw_input is None:
        return None

    try:
        return json.loads(raw_input)
    except json.JSONDecodeError:
        return raw_input


def _build_pipeline_context(base_dir: Path) -> Dict[str, Any]:
    print(f"connecting to MongoDB: {MONGO_URI}")
    print(f"using database: {MONGO_DB_NAME}")

    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]

    return {
        "base_dir": str(base_dir),
        "mongo_client": mongo_client,
        "db": mongo_db,
    }


def run_pipeline(initial_input: Any = None, pipeline_dir: str | Path | None = None) -> Tuple[Any, int, int]:
    base_dir = Path(pipeline_dir) if pipeline_dir else Path(__file__).parent
    context = _build_pipeline_context(base_dir)

    print(f"pipeline directory: {base_dir}")
    print(f"configured steps: {len(PIPELINE_STEPS)}")

    output_data = initial_input
    attempted_steps = 0
    executed_steps = 0

    try:
        for step_name, script_name in PIPELINE_STEPS:
            attempted_steps += 1

            previous_output = output_data
            output_data = _run_single_step(
                step_name=step_name,
                script_name=script_name,
                input_data=output_data,
                context=context,
                base_dir=base_dir,
            )

            if output_data is not previous_output:
                executed_steps += 1
    finally:
        mongo_client = context.get("mongo_client")
        if mongo_client is not None:
            mongo_client.close()
            print("mongodb connection closed")

    print(f"pipeline finished: executed contracts {executed_steps}/{attempted_steps}")
    return output_data, attempted_steps, executed_steps


def _print_result(result: Any) -> None:
    print("\nresult payload:")
    try:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except TypeError:
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="run fetch/preprocess/sentiment/geospatial pipeline")
    parser.add_argument(
        "--pipeline-dir",
        default=str(Path(__file__).parent),
        help="directory containing step scripts",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="optional initial input (json string preferred)",
    )

    args = parser.parse_args()
    initial_input = _parse_initial_input(args.input)
    result, attempted_steps, executed_steps = run_pipeline(
        initial_input=initial_input,
        pipeline_dir=args.pipeline_dir,
    )

    print(f"steps summary: attempted={attempted_steps} | executed={executed_steps}")
    _print_result(result)


if __name__ == "__main__":
    main()
