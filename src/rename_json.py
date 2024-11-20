import json
import os

for _, _, fn_list in os.walk("fbd"):
    for fn in fn_list:
        if fn.endswith(".json"):
            a = json.load(open(f"fbd/{fn}", "r"))
            id = a['id']
            os.rename(f"fbd/{fn}", f"fbd/{id}.json")
