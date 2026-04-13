import json, random

with open("data/finance_sft_gpt_cleaned.json", encoding="utf-8") as f:
    cleaned = json.load(f)
with open("data/finance_sft_new_generated.json", encoding="utf-8") as f:
    generated = json.load(f)

all_data = cleaned + generated

# 去重
seen = set()
unique = []
for d in all_data:
    key = d["instruction"].strip()
    if key not in seen:
        seen.add(key)
        unique.append(d)

random.shuffle(unique)

with open("data/finance_sft_final.json", "w", encoding="utf-8") as f:
    json.dump(unique, f, ensure_ascii=False, indent=2)

inst_lens = [len(d["instruction"]) for d in unique]
out_lens = [len(d["output"]) for d in unique]
print(f"GPT清洗保留: {len(cleaned)}条")
print(f"Claude补充生成: {len(generated)}条")
print(f"去重后最终: {len(unique)}条")
print(f"instruction平均长度: {sum(inst_lens)//len(inst_lens)}字")
print(f"output平均长度: {sum(out_lens)//len(out_lens)}字")
