from openai import AsyncOpenAI
import asyncio, json, re, os

client = AsyncOpenAI() 

CONCURRENCY = 5  

SYSTEM_PROMPT = """你是一个A股投资实战教练，专门解答中国散户投资者的真实问题。

【生成规则 - 必须严格遵守】

一、关于问题（instruction）：
- 10-40字，口语化，模拟真实散户提问
- 句式必须多样化，均匀使用以下开头：
  "为什么…" "什么情况下…" "…是什么信号" "…和…有什么区别" "…说明什么" "怎么判断…" "该不该…" "…靠谱吗" "…怎么回事" "看到…该怎么办"
- 禁止连续两条用相同句式开头

二、关于回答（output）：
- 150-400字
- 【禁止回声开头】不要用"当股价…时""关于…""面对…的情况"重复问题，直接给结论
- 【必须有具体数字】每条回答至少包含2个具体数字或判断标准（如"换手率超过15%""PE低于25倍""跌破5日均线3%以上"）
- 【禁止空洞建议】不要说"建议结合其他指标综合判断""需要谨慎""保持关注"，要说具体看什么、怎么看、阈值是多少
- 回答结构：直接结论（1-2句）→ 判断方法/逻辑（含具体数据）→ 操作建议（含具体条件）
- 风格：像一个赚过钱也亏过钱的老股民在教新手
- 纯文本，不要任何markdown格式

三、质量红线（触犯任何一条该条作废）：
- 回答中没有任何具体数字 → 作废
- 回答开头复述问题 → 作废
- 出现"投资有风险，入市需谨慎" → 作废

【好的示例】
Q: 一只股票连续缩量三天还在跌，是不是主力在洗盘？
A: 不一定，缩量下跌有两种可能。如果股价在前期涨幅30%以内，日换手率从5%以上降到2%以下，同时没有跌破20日均线，大概率是洗盘，主力不愿意在低位放量出货。但如果股价已经翻倍以上，即使缩量下跌也可能是出货尾声——散户都不敢接盘了自然缩量。你可以看筹码分布，如果上方套牢盘不大，下方获利盘集中且没松动，洗盘概率更大。操作上，可以在缩量到地量（换手率低于1%）时小仓位试探，止损设在近期低点下方3%。

Q: 为什么有些股票业绩很好但股价就是不涨？
A: 这种情况很常见，业绩好不等于股价涨，关键看预期差。如果一只股票净利润增长30%，但市场之前已经预期它增长40%，公布业绩反而是利空。另外要看估值水平，如果PE已经50倍以上，即使业绩不错，资金也不愿意在高位继续推。还有一种情况是大股东或机构在减持，你可以查公告看最近半年内是否有5%以上股东减持计划。判断方法：看业绩公布前后的股价反应，如果利好不涨反跌，说明市场预期已经打满，短期不要追。

请严格按以下JSON格式返回10条数据，不要包含任何其他文字：
[{"instruction":"问题","input":"","output":"回答"},...]"""

categories = {
    "技术指标实战": {
        "prompt": "围绕K线形态（锤子线、十字星、吞没、早晨之星、仙人指路等）、均线系统（金叉死叉、多空排列、均线粘合）、MACD（背离、零轴、柱状缩放）、RSI/KDJ/布林带、成交量异动、分时图看盘生成问答。问题必须场景化，如'MACD底背离了但股价还在跌是怎么回事'而不是'MACD怎么用'"
    },
    "选股方法": {
        "prompt": "围绕基本面筛选（PE/PB/ROE/现金流/毛利率）、技术面选股（突破、放量启动、底部形态）、题材选股（政策驱动、行业周期）、不同市场环境的选股策略生成问答。问题如'ROE高但负债率也高的公司能买吗'"
    },
    "买卖时机": {
        "prompt": "围绕买入时机（回调确认、突破追入、底部试探）、卖出时机（目标位、破位、情绪见顶）、加减仓节奏、左侧vs右侧交易、追涨vs等回调生成问答。问题如'股票涨了20%开始放量滞涨，是该走还是再等等'"
    },
    "仓位与风控": {
        "prompt": "围绕单只仓位上限、总仓位控制、止损方法（比例/技术位/时间）、止盈策略（目标位/移动/分批）、回撤控制、分散投资生成问答。问题如'亏了15%不想止损因为觉得能涨回来，这种心态对吗'"
    },
    "行业板块分析": {
        "prompt": "围绕半导体、AI算力、新能源（光伏/锂电/风电）、医药、消费、军工、金融地产、汽车、传媒等板块的投资逻辑、板块轮动、产业链关系、龙头识别生成问答。问题如'AI板块已经涨了一年了，现在上车是不是太晚了'"
    },
    "财报分析": {
        "prompt": "围绕三大报表核心指标、业绩预告解读、同比环比分析、商誉减值风险、关联交易识别、应收账款异常、存贷双高生成问答。问题如'一家公司营收增长但扣非净利润下降，是什么原因'"
    },
    "大盘与资金面": {
        "prompt": "围绕指数走势判断、北向资金解读、融资余额、市场情绪（涨跌比/涨停数/成交额）、政策影响（降准降息/印花税）、板块轮动风格切换生成问答。问题如'北向资金连续流出一周了，会不会引发大跌'"
    },
    "交易规则与制度": {
        "prompt": "围绕涨跌停制度（主板/科创/创业板/北交所）、集合竞价、打新规则、可转债、ST退市、注册制、融资融券、龙虎榜、大宗交易生成问答。问题如'集合竞价最后突然拉高是什么意思，能跟吗'"
    },
    "投资心理与纪律": {
        "prompt": "围绕追涨杀跌心理、锚定效应、沉没成本、过度交易、亏损后心态崩塌、交易计划执行生成问答。问题如'每次一止损股票就涨回来，以后还要不要设止损'"
    },
    "基金与资产配置": {
        "prompt": "围绕指数基金vs主动基金、ETF套利、定投策略、股债配置、基金评价指标（夏普/回撤/alpha）生成问答。问题如'定投亏了两年了还要不要继续，什么时候能回本'"
    },
}

STATE_FILE = "data/finance_sft_generated_state.json"
OUTPUT_FILE = "data/finance_sft_new_generated.json"

FILLER_PHRASES = [
    "需要谨慎", "谨慎对待", "保持谨慎", "结合其他指标", "综合判断",
    "综合分析", "保持关注", "密切关注", "持续关注", "根据自身情况",
    "根据个人风险偏好", "投资有风险", "入市需谨慎"
]


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"all_generated": [], "seen_instructions": [], "category_progress": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def validate_and_clean(item, seen_instructions):
    """校验单条数据，通过返回清洗后的dict，否则返回None。纯函数，不修改seen_instructions。"""
    if not (isinstance(item, dict) and "instruction" in item and "output" in item):
        return None
    inst = item["instruction"].strip()
    out = item["output"].strip()

    if len(inst) < 8 or len(inst) > 60:
        return None
    if len(out) < 100 or len(out) > 500:
        return None
    if inst in seen_instructions:
        return None

    for seen in seen_instructions:
        common = len(set(inst) & set(seen))
        sim = common / max(len(set(inst)), len(set(seen)), 1)
        if sim > 0.8 and abs(len(inst) - len(seen)) < 5:
            return None

    if not re.search(r'\d+[%％倍日天周月年元万亿]', out):
        return None

    if sum(1 for f in FILLER_PHRASES if f in out) >= 3:
        return None

    out = re.sub(r'#{1,4}\s*', '', out)
    out = re.sub(r'\*\*(.*?)\*\*', r'\1', out)
    out = re.sub(r'^- ', '', out, flags=re.MULTILINE)

    return {"instruction": inst, "input": "", "output": out}


async def fetch_batch(cat_name, cat_prompt, existing_questions):
    existing_summary = ""
    if existing_questions:
        recent = existing_questions[-30:]
        existing_summary = "\n\n【已生成过的问题，不要生成类似的】\n" + "\n".join(f"- {q}" for q in recent)

    for retry in range(3):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"主题：{cat_name}\n要求：{cat_prompt}{existing_summary}\n\n请生成10条不重复的问答数据。"}
                ],
                temperature=1.0,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```json")[-1].split("```")[0] if "```json" in text else text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            print(f"    [{cat_name}] 重试{retry+1}/3: {e}")
            await asyncio.sleep(2)
    return []


async def generate_category(cat_name, cat_prompt, cat_target, already_done,
                             seen_instructions, all_generated, category_progress,
                             state, save_lock):
    existing_questions = [
        d["instruction"] for d in all_generated
        if d.get("_category") == cat_name
    ]
    cat_data_count = already_done
    batch_round = 0
    max_rounds = (cat_target // 10 + 5) * 3

    print(f">> {cat_name}: target={cat_target}, done={already_done}, need={cat_target - already_done}")

    while cat_data_count < cat_target and batch_round < max_rounds:
        batch_round += 1
        tasks = [
            fetch_batch(cat_name, cat_prompt, list(existing_questions))
            for _ in range(CONCURRENCY)
        ]
        results = await asyncio.gather(*tasks)

        round_valid = []
        for items in results:
            for item in items:
                if cat_data_count + len(round_valid) >= cat_target:
                    break
                cleaned = validate_and_clean(item, seen_instructions)
                if cleaned is None:
                    continue
                inst = cleaned["instruction"]
                if inst in seen_instructions:
                    continue
                seen_instructions.add(inst)
                existing_questions.append(inst)
                full_item = {**cleaned, "_category": cat_name}
                round_valid.append(full_item)

        all_generated.extend(round_valid)
        cat_data_count += len(round_valid)
        category_progress[cat_name] = cat_data_count

        async with save_lock:
            state["all_generated"] = all_generated
            state["seen_instructions"] = list(seen_instructions)
            state["category_progress"] = category_progress
            save_state(state)

        print(f"  [{cat_name}] 轮次{batch_round}×{CONCURRENCY}并发 | 本轮有效{len(round_valid)}条 | 累计{cat_data_count}/{cat_target}")

    print(f"[done] {cat_name}: {cat_data_count}\n")


async def main():
    with open("data/finance_sft_gpt_cleaned.json", encoding="utf-8") as f:
        cleaned = json.load(f)
    cleaned_count = len(cleaned)
    target_total = 5000
    need = target_total - cleaned_count
    per_category = need // 10
    extra = need % 10

    print(f"清洗后已有: {cleaned_count} 条")
    print(f"目标总量: {target_total} 条")
    print(f"需要补生成: {need} 条")
    print(f"每类别约: {per_category} 条（前{extra}个类别多1条）")
    print(f"并发数: {CONCURRENCY} 请求/轮\n")

    state = load_state()
    all_generated = state["all_generated"]
    seen_instructions = set(state["seen_instructions"])
    category_progress = state["category_progress"]

    for d in cleaned:
        seen_instructions.add(d["instruction"].strip())

    save_lock = asyncio.Lock()

  
    for idx, (cat_name, cat_config) in enumerate(categories.items()):
        cat_target = per_category + (1 if idx < extra else 0)
        already_done = category_progress.get(cat_name, 0)
        if already_done >= cat_target:
            print(f"[skip] {cat_name}: already {already_done}/{cat_target}\n")
            continue

        await generate_category(
            cat_name, cat_config["prompt"], cat_target, already_done,
            seen_instructions, all_generated, category_progress,
            state, save_lock
        )

    output = [
        {"instruction": d["instruction"], "input": d["input"], "output": d["output"]}
        for d in all_generated
    ]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n补充生成完成: {len(output)}条 → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
