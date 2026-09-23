#!/usr/bin/env python3
"""SemIf 语义决策客户端(llama-server /v1/chat/completions + top_logprobs 读出).

复刻 theoleecj/semif 的 direct 模式语义:
  prompt = system(DIRECT_SYSTEM) + user(json{evidence,criterion,options[letter...]})
  读助手首个 token 位置的 A/B/C 字母 logprob → softmax = 选项概率
(browser webgpu-demo worker.js 同款读出;字母几乎必在 top-20,近似=精确)

Jev 兼容三原语 + 电池(响应字段对齐 api.typesafe.ai/v1/systemone 实测 schema):
  choice(state, instructions, criteria={id:描述})        -> {"type","choice","confidence","probabilities"}
  score(state, instructions, criteria=[低..高等级描述])  -> {"type","score","confidence","legend","probabilities"}
  noul(state, instructions)                              -> {"type","noul"}
  ask(state, questions={名:{type,instructions,criteria}})-> {"model","answers","usage"}
用法: semif_client.py --input examples/decisions.jsonl [--endpoint http://127.0.0.1:8084]
"""
import argparse, json, math, sys, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")
_NOUL_LETTERS = ("A", "N")  # 双字母读出:A=肯定/N=否定

def _letters_probs(state, instructions, options, endpoint, timeout=120, stats=None, letters=None):
    """共享读出核心:DIRECT_SYSTEM + user(json{evidence,criterion,options}),
    max_tokens=1 + temperature=0 + enable_thinking=False + top_logprobs=20,
    字母 token 精确匹配、先到先得(防 "A"/" A" 双胞胎覆盖)。
    options: list[(id, description)];letters: 实际出示的字母(默认按位 A/B/C...);
    返回 {id: 概率}(top-20 缺席的字母记 0.0);stats 非 None 时回填 seconds/prompt_tokens。"""
    if letters is None:
        letters = LETTERS[:len(options)]
    payload = {
        "evidence": state,
        "criterion": instructions,
        "options": [{"letter": letters[i], "description": desc} for i, (_, desc) in enumerate(options)],
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
    # ⚠ 精确匹配且先到先得:词表里有多个字符串为"A"的 token(如" A"),strip 归一会互相覆盖拿错 logprob
    lp = {}
    for e in tops:
        tok = e["token"]
        if tok in letters and tok not in lp:
            lp[tok] = e["logprob"]
    if not lp:
        raise RuntimeError(f"letters {letters} not in top tokens: {[e['token'] for e in tops[:8]]}")
    mx = max(lp.values())
    weights = {L: math.exp(v - mx) for L, v in lp.items()}
    total = sum(weights.values())
    by_letter = {L: w / total for L, w in weights.items()}
    if stats is not None:
        stats.update({"seconds": round(wall, 2),
                      "prompt_tokens": r.get("usage", {}).get("prompt_tokens") or 0,
                      "letters_missing": [L for L in letters if L not in lp]})
    return {oid: by_letter.get(letters[i], 0.0) for i, (oid, _) in enumerate(options)}

def score_row(row, endpoint, timeout=120):
    try:
        stats = {}
        probs = _letters_probs(row["state"], row["question"],
                               [(o["id"], o["description"]) for o in row["options"]],
                               endpoint, timeout, stats)
    except RuntimeError as e:
        return {"id": row["id"], "error": str(e)}
    return {
        "id": row["id"],
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "argmax": max(probs, key=probs.get),
        "letters_missing_from_top20": stats["letters_missing"],
        "seconds": stats["seconds"],
        "usage": stats["prompt_tokens"],
    }

def _margin(probs):
    v = sorted(probs.values(), reverse=True)
    return v[0] - (v[1] if len(v) > 1 else 0.0)

def choice(state, instructions, criteria, endpoint, timeout=120, _stats=None):
    """Jev choice:criteria={选项id: 描述}(顺序无关);返回 argmax 选项、
    confidence=top1-top2 概率差、全选项概率。"""
    options = [(k, criteria[k]) for k in criteria]
    stats = _stats if _stats is not None else {}
    probs = _letters_probs(state, instructions, options, endpoint, timeout, stats)
    return {
        "type": "choice",
        "choice": max(probs, key=probs.get),
        "confidence": round(_margin(probs), 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    }

def score(state, instructions, criteria, endpoint, timeout=120, _stats=None):
    """Jev score:criteria=有序等级描述数组(低→高,2-10 级),按序映射 A/B/C...;
    score = Σ(i·p_i),0..n-1 浮点,可落在整级之间;legend/probabilities 以等级下标字符串为键。"""
    if not 2 <= len(criteria) <= 10:
        raise ValueError(f"score criteria needs 2-10 ordered levels, got {len(criteria)}")
    options = [(str(i), d) for i, d in enumerate(criteria)]
    stats = _stats if _stats is not None else {}
    probs = _letters_probs(state, instructions, options, endpoint, timeout, stats)
    val = sum(i * p for i, p in enumerate(probs.values()))
    return {
        "type": "score",
        "score": round(val, 4),
        "confidence": round(_margin(probs), 4),
        "legend": {str(i): d for i, d in enumerate(criteria)},
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    }

def noul(state, instructions, endpoint, timeout=120, criteria=None, _stats=None):
    """Jev noul:双字母 A/N 读出,noul = P(A)/(P(A)+P(N)),0..1,越大越肯定;
    criteria 可选 {true: 描述, false: 描述} 用于定制 A/N 选项描述。"""
    if isinstance(criteria, dict) and criteria.get("true") and criteria.get("false"):
        options = [("A", criteria["true"]), ("N", criteria["false"])]
    else:
        options = [("A", "Yes - it holds true for the evidence."),
                   ("N", "No - it does not hold true for the evidence.")]
    stats = _stats if _stats is not None else {}
    probs = _letters_probs(state, instructions, options, endpoint, timeout, stats, _NOUL_LETTERS)
    a, n = probs.get("A", 0.0), probs.get("N", 0.0)
    return {"type": "noul", "noul": round(a / (a + n), 4) if (a + n) > 0 else 0.0}

_MODEL_CACHE = {}
def _model_name(endpoint, timeout=30):
    if endpoint not in _MODEL_CACHE:
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=timeout) as r:
                _MODEL_CACHE[endpoint] = json.load(r)["data"][0]["id"]
        except Exception:
            _MODEL_CACHE[endpoint] = "semif-local"
    return _MODEL_CACHE[endpoint]

def ask(state, questions, endpoint, timeout=180):
    """Jev systemone 电池:questions={名:{"type":choice|score|noul,"instructions":...,"criteria":...}},
    逐题分派到三原语;evidence 前缀字节级一致,吃 llama.cpp KV 前缀缓存;
    返回 {"model","answers","usage"}(output_tokens 恒 0,对齐 Jev 读出侧计费口径)。"""
    answers, input_tokens = {}, 0
    for name, q in questions.items():
        t = q.get("type")
        stats = {}
        try:
            if t == "choice":
                a = choice(state, q["instructions"], q["criteria"], endpoint, timeout, _stats=stats)
            elif t == "score":
                a = score(state, q["instructions"], q["criteria"], endpoint, timeout, _stats=stats)
            elif t == "noul":
                a = noul(state, q.get("instructions"), endpoint, timeout,
                         criteria=q.get("criteria"), _stats=stats)
            else:
                a = {"type": t, "error": f"unknown question type: {t!r}"}
        except RuntimeError as e:
            a = {"type": t, "error": str(e)}
        answers[name] = a
        input_tokens += stats.get("prompt_tokens", 0)
    return {
        "model": _model_name(endpoint),
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
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
