"""
Contract required by pipeline:
1) File is executed by runpy from pipeline.py
2) Must expose: run(input_data, context) -> output_data
3) input_data = previous step output
4) context includes:
   - base_dir: str
   - mongo_client: MongoClient (shared for whole pipeline run)
   - db: pymongo Database (shared)
5) Return output_data for next step
"""
import json
from typing import Any, Dict

STEP_NAME = "replace_with_step_name"

def run(input_data: Any, context: Dict[str, Any]) -> Any:
    """
    Execute one pipeline step.
    Args:
        input_data: Payload from previous step. Can be any Python type.
        context: Shared runtime context injected by pipeline.py.
    Returns:
        Any: Payload to pass to the next pipeline step.
    """
    print(f"running step: {STEP_NAME}")
    
    # Shared DB objects from pipeline (do NOT create new MongoClient here)
    db = context.get("db")
    mongo_client = context.get("mongo_client")  # usually not needed directly
    
    # TODO: implement your step logic
    
    # Example passthrough-style output:
    output_data = {
        "step": STEP_NAME,
        "status": "ok",
        "input_type": type(input_data).__name__,
        "data": input_data,
    }
    
    print(f"step completed: {STEP_NAME}")
    return output_data

if __name__ == "__main__":
    # Local standalone test only (not used by pipeline run)
    demo_input = {"demo": True}
    demo_context = {"base_dir": ".", "db": None, "mongo_client": None}
    demo_output = run(demo_input, demo_context)
    
    print("local test output:")
    try:
        print(json.dumps(demo_output, indent=2, ensure_ascii=True))
    except TypeError:
        print(demo_output)
        