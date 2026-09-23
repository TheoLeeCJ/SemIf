#!/usr/bin/env python3
"""SemIf 语义决策客户端(llama-server /v1/chat/completions + top_logprobs 读出).

复刻 theoleecj/semif 的 direct 模式语义:
  prompt = system(DIRECT_SYSTEM) + user(json{evidence,criterion,options[letter...]})
  读助手首个 token 位置的 A/B/C 字母 logprob → softmax = 选项概率
(browser webgpu-demo worker.js 同款读出;字母几乎必在 top-20,近似=精确)
用法: semif_client.py --input examples/decisions.jsonl [--endpoint http://127.0.0.1:8084]
"""
import argparse, json, math, sys, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")

def score_row(row, endpoint, timeout=120):
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(row["options"])],
    }
    body = json.dumps({
        "messages": [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": 1, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True, "top_logprobs": 20,
    }).encode()
    req = urllib.request.Request(f"{endpoint}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    wall = time.perf_counter() - t0
    tops = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    letters = LETTERS[:len(row["options"])]
    # ⚠ 精确匹配且先到先得:词表里有多个字符串为"A"的 token(如" A"),strip 归一会互相覆盖拿错 logprob
    lp = {}
    for e in tops:
        tok = e["token"]
        if tok in letters and tok not in lp:
            lp[tok] = e["logprob"]
    picked = {L: lp[L] for L in letters if L in lp}
    if not picked:
        return {"id": row["id"], "error": f"letters {letters} not in top tokens: {list(lp)[:8]}"}
    mx = max(picked.values())
    weights = {L: math.exp(v - mx) for L, v in picked.items()}
    total = sum(weights.values())
    probs = {L: w / total for L, w in weights.items()}
    missing = [L for L in letters if L not in lp]
    return {
        "id": row["id"],
        "probabilities": {row["options"][i]["id"]: round(probs.get(LETTERS[i], 0.0), 4)
                          for i in range(len(row["options"]))},
        "argmax": max(probs, key=probs.get),
        "letters_missing_from_top20": missing,
        "seconds": round(wall, 2),
        "usage": r.get("usage", {}).get("prompt_tokens"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="examples/decisions.jsonl")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8084")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input) if l.strip()]
    for row in rows:
        print(json.dumps(score_row(row, a.endpoint), ensure_ascii=False))

if __name__ == "__main__":
    main()
