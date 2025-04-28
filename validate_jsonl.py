import json

PATH = "src/data/synthetic_state_prompt_ext.jsonl"

bad = []
with open(PATH, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad.append((lineno, "JSON decode error"))
            continue
        # check for exactly the keys you need
        if "state" not in obj or "prompt" not in obj:
            bad.append((lineno, list(obj.keys())))

if not bad:
    print("✅ All lines look good!")
else:
    print(f"❌ Found {len(bad)} bad lines:")
    for ln, issue in bad:
        print(f"  • Line {ln}: {issue}")
