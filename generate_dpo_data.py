from openai import AsyncOpenAI
import json, asyncio, random, re, os

client = AsyncOpenAI()

SYSTEM_PROMPT = """你是一个A股投资领域的数据标注专家，需要为每个投资问题生成一个"好回答"和一个"差回答"。

【好回答（chosen）的标准】
1. 直接回答问题，开头就给结论，不绕弯子
2. 包含至少2个具体数字或判断标准（如"PE低于25倍""换手率超过10%""跌破支撑位3%止损"）
3. 有实际可操作的建议（具体怎么做、什么条件下做）
4. 像有经验的老股民在教新手，语气自然
5. 150-300字

【差回答（rejected）的标准 — 要像真实的差回答，不能太离谱】
差回答不是胡说八道，而是那种"说了等于没说"的回答。具体表现为以下几种（每条随机选1-2种）：
1. 废话型：全是"建议关注""需要谨慎""综合判断"，没有任何具体数字或操作建议
2. 回声型：先花30字复述问题，然后才开始回答
3. 教科书型：只解释概念定义，不给实战建议
4. 万金油型：回答能套用到任何问题上，毫无针对性
5. 免责型：每句话都在说风险，"投资有风险""不构成投资建议"
6. 100-200字（比好回答短，内容空洞）

【重要】差回答要逼真，不要写成明显的错误答案。

请严格按以下JSON格式返回5组数据，不要包含其他文字：
[{"prompt":"问题","chosen":"好回答","rejected":"差回答"},...]"""


async def generate_batch(questions: list[str]) -> list[dict]:
    """一次生成 5 组偏好对，失败最多重试 3 次。"""
    q_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    for retry in range(3):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"请为以下5个问题各生成一组偏好对：\n{q_list}"},
                ],
                temperature=0.9,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            items = json.loads(text)

            valid = []
            for item in items:
                if not all(k in item for k in ["prompt", "chosen", "rejected"]):
                    continue
                chosen   = item["chosen"].strip()
                rejected = item["rejected"].strip()
                # chosen 必须含具体数字
                if not re.search(r'\d+[%％倍日天周月年元万亿]', chosen):
                    continue
                # chosen 要比 rejected 长（内容更丰富）
                if len(chosen) <= len(rejected):
                    continue
                # chosen 至少 100 字
                if len(chosen) < 100:
                    continue
                item["chosen"]   = chosen
                item["rejected"] = rejected
                valid.append(item)
            return valid

        except Exception as e:
            print(f"  重试{retry+1}/3: {e}")
            await asyncio.sleep(2)
    return []


async def main():
    with open("data/finance_sft_final.json", encoding="utf-8") as f:
        sft_data = json.load(f)

    all_instructions = [d["instruction"] for d in sft_data]
    random.seed(42)
    random.shuffle(all_instructions)
    selected = all_instructions[:2000]           # ← 2000 条

    batches = [selected[i:i+5] for i in range(0, len(selected), 5)]  # 400 批

    all_dpo_data: list[dict] = []
    semaphore = asyncio.Semaphore(5)             # 并发 5 个请求

    async def process_batch(batch_idx: int, questions: list[str]):
        async with semaphore:
            result = await generate_batch(questions)
            if result:
                all_dpo_data.extend(result)
            if (batch_idx + 1) % 20 == 0:
                print(f"  进度: {batch_idx+1}/{len(batches)} 批 | 已生成 {len(all_dpo_data)} 组偏好对")
            await asyncio.sleep(0.5)

    print(f"共 {len(batches)} 批，每批 5 条，并发 5 ...")
    await asyncio.gather(*[process_batch(i, b) for i, b in enumerate(batches)])

    random.shuffle(all_dpo_data)
    os.makedirs("data", exist_ok=True)
    with open("data/finance_dpo_data.json", "w", encoding="utf-8") as f:
        json.dump(all_dpo_data, f, ensure_ascii=False, indent=2)

    chosen_lens   = [len(d["chosen"])   for d in all_dpo_data]
    rejected_lens = [len(d["rejected"]) for d in all_dpo_data]
    print(f"\n=== 生成完成 ===")
    print(f"总偏好对数      : {len(all_dpo_data)}")
    print(f"chosen 平均长度 : {sum(chosen_lens)  // len(chosen_lens)}字")
    print(f"rejected 平均长度: {sum(rejected_lens) // len(rejected_lens)}字")

    print("\n=== 样例（前3条）===")
    for d in all_dpo_data[:3]:
        print(f"\nQ: {d['prompt']}")
        print(f"[chosen]   {d['chosen'][:150]}...")
        print(f"[rejected] {d['rejected'][:150]}...")
        print("-" * 50)


asyncio.run(main())
