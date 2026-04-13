"""
DB-Focused Step Template
Use this variant if your step primarily involves reading from or writing to MongoDB.
"""

from typing import Any, Dict

STEP_NAME = "replace_with_step_name"

def run(input_data: Any, context: Dict[str, Any]) -> Any:
    db = context["db"]  # shared DB from pipeline
    collection = db["replace_with_collection_name"]
    
    print(f"running step: {STEP_NAME}")
    
    # Example read
    docs = list(collection.find({}, {"_id": 1}).limit(5))
    
    # Example output for next step
    output_data = {
        "step": STEP_NAME,
        "count": len(docs),
        "docs": docs,
        "previous": input_data,
    }
    
    print(f"step completed: {STEP_NAME} | docs={len(docs)}")
    return output_data