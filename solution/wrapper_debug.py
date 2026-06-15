"""Debug wrapper to trace private test errors."""
from solution.wrapper import mitigate as original_mitigate
import os
import json

DEBUG_LOG = "debug_wrapper.log"

def mitigate_debug(call_next, question, config, context):
    """Wrapper with detailed debugging."""
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== Request {context.get('qid')} ===\n")
        f.write(f"Question: {question[:100]}\n")
        f.write(f"Config model: {config.get('model')}\n")
        f.write(f"Config provider: {config.get('provider')}\n")
    
    try:
        result = original_mitigate(call_next, question, config, context)
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"Status: {result.get('status')}\n")
            f.write(f"Answer: {result.get('answer', '')[:50]}\n")
            if result.get('meta'):
                meta = result['meta']
                f.write(f"Meta error: {meta.get('error')}\n")
                f.write(f"Meta model: {meta.get('model')}\n")
        return result
    except Exception as exc:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"Exception: {type(exc).__name__}: {exc}\n")
            import traceback
            f.write(traceback.format_exc())
        raise

# Replace mitigate
if __name__ == "__main__":
    print("This is a debug wrapper module")
