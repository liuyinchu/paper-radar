from __future__ import annotations
import os
import time
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, InternalServerError
import arxiv

# ---------- 常量与缓存 ----------
CACHE_DIR = Path(".paperscout_cache")
SUM_CACHE = CACHE_DIR / "sum_cache"
SUM_CACHE.mkdir(parents=True, exist_ok=True)

# ---------- 工具函数 ----------

def _read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    ids = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            ids.append(s)
    return ids

def _strip_version(arxiv_id: str) -> str:
    s = arxiv_id.strip()
    i = s.rfind("v")
    if i != -1 and i < len(s) - 1 and s[i+1:].isdigit():
        return s[:i]
    return s

def _normalize_id(s: str, ignore_version: bool) -> str:
    return _strip_version(s) if ignore_version else s

def _today_filename(prefix: str = "", suffix: str = "_Daily_Arxiv.json") -> str:
    from datetime import datetime
    # 本地时区日期 -> DD_MM_YY
    return f"{prefix}{datetime.now().strftime('%d_%m_%y')}{suffix}"

# ---------- 载入与对齐当前 run 的元数据 ----------

@dataclass
class PaperMeta:
    arxiv_id: str          # 原始带版本
    date: str
    title: str
    pdf: str
    authors: List[str]
    theme: str
    abstract: Optional[str] = None
    primary_category: Optional[str] = None
    categories: Optional[List[str]] = None

def _load_results(results_json: Path) -> List[PaperMeta]:
    if not results_json.exists():
        raise FileNotFoundError(f"results file not found: {results_json}")
    raw = json.loads(results_json.read_text(encoding="utf-8"))
    out: List[PaperMeta] = []
    for it in raw:
        # 兼容此前两种 key：arXiv_id 或 arXiv id（带空格）
        arxid = it.get("arXiv_id") or it.get("arXiv id")
        if not arxid:
            continue
        out.append(PaperMeta(
            arxiv_id=arxid,
            date=it.get("date",""),
            title=it.get("title","").strip(),
            pdf= it.get("arXiv_pdf_adress") or it.get("arXiv pdf adress",""),
            authors=list(it.get("author_list") or []),
            theme=it.get("theme",""),
            abstract=it.get("abstract"),
            primary_category=it.get("primary_category"),
            categories=it.get("categories"),
        ))
    return out

# ---------- 如缺摘要，则从 arXiv 批量补齐（合规：单连接+≥3s/次） ----------

def _fetch_abstracts_for_ids(id_batch: List[str]) -> Dict[str, str]:
    """
    用 id_list 一次性取一批条目，返回 {带版本id: 摘要}。
    """
    if not id_batch:
        return {}
    # arxiv.Search 支持 id_list；client 负责节流与翻页（这里不会翻页）
    search = arxiv.Search(id_list=id_batch)
    client = arxiv.Client(page_size=len(id_batch), delay_seconds=3.1, num_retries=3)
    out = {}
    for e in client.results(search):
        aid = e.entry_id.split("/")[-1]
        out[aid] = (e.summary or "").strip()
    return out

def ensure_abstracts(papers: List[PaperMeta], ignore_version: bool, max_batch: int = 50) -> None:
    # 找出缺摘要的
    need: List[PaperMeta] = [p for p in papers if not p.abstract]
    if not need:
        return
    # 批量以“带版本 id”为键；为了兼容对齐，也记录“无版本”键
    todo_ids = [p.arxiv_id for p in need]
    # 分批请求
    for i in range(0, len(todo_ids), max_batch):
        batch = todo_ids[i:i+max_batch]
        abs_map = _fetch_abstracts_for_ids(batch)
        # 回填
        for p in need:
            if p.abstract:
                continue
            # 优先带版本精确命中
            if p.arxiv_id in abs_map:
                p.abstract = abs_map[p.arxiv_id]
            else:
                # 尝试无版本匹配
                nv = _strip_version(p.arxiv_id)
                # abs_map 的 key 都是带版本，如果作者只传了 v1，可能对不上；尽力而为
                for k, v in abs_map.items():
                    if _strip_version(k) == nv:
                        p.abstract = v
                        break

# ---------- DeepSeek 调用封装（不写提示词，读文件/字符串） ----------

@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str = "deepseek-chat"
    timeout: float = 60.0
    max_retries: int = 5
    min_interval: float = 0.8  # 调用间隔，避免打满

def _load_prompt_text(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()

def _client_from_env(cfg: LLMConfig) -> OpenAI:
    # 允许通过环境变量覆盖 base_url
    base_url = os.environ.get("OPENAI_BASE_URL", cfg.base_url)
    key = os.environ.get("DEEPSEEK_API_KEY", cfg.api_key)
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing. Export it or pass --api-key.")
    return OpenAI(api_key=key, base_url=base_url, timeout=cfg.timeout)

def _cache_path_for(aid: str) -> Path:
    # 基于 arXiv id（带版本）缓存
    safe = aid.replace("/", "_")
    return SUM_CACHE / f"{safe}.json"

def _save_cache(aid: str, obj: Dict) -> None:
    _cache_path_for(aid).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_cache(aid: str) -> Optional[Dict]:
    p = _cache_path_for(aid)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def call_llm_for_paper(client: OpenAI, model: str, system_prompt: str, paper: PaperMeta) -> Dict[str, str]:
    """
    实际请求：你负责提供 system_prompt；我把论文信息以结构化 JSON 放进 user message。
    期望模型返回 problem/method/result 的 JSON 字符串；解析失败则直接包一层。
    """
    # 优先用缓存
    cached = _load_cache(paper.arxiv_id)
    if cached and all(k in cached for k in ("problem","method","result")):
        return cached

    # 组装用户侧内容（不写提示词，只提供素材）
    user_payload = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract or "",
        "authors": paper.authors,
        "pdf_url": paper.pdf,
        "theme": paper.theme,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        # 你的提示词可以指导模型“严格输出 JSON，包含 problem/method/result”
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

    # 简单重试 + 指数回退
    backoff = 1.2
    delay = 1.0
    for attempt in range(1, 1 + 5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            txt = resp.choices[0].message.content.strip()
            # 解析 JSON
            data = None
            try:
                data = json.loads(txt)
            except Exception:
                # 非 JSON：兜底塞到 result，把 problem/method 置空
                data = {"problem":"", "method":"", "result": txt}
            # 仅保留三字段（其它丢弃）
            out = {
                "problem": str(data.get("problem","")).strip(),
                "method":  str(data.get("method","")).strip(),
                "result":  str(data.get("result","")).strip(),
            }
            _save_cache(paper.arxiv_id, out)
            # 节流
            time.sleep(0.8)
            return out
        except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
            if attempt >= 5:
                # 最终失败，给出空模板，避免中断整批
                return {"problem":"", "method":"", "result": f"[LLM error: {type(e).__name__}]"}
            time.sleep(delay)
            delay *= backoff

# ---------- 主流程：读取纯新ID → 对齐元数据 →（必要时补摘要）→ 调模型 → 组装目标JSON ----------

def summarize_pipeline(
    new_ids_file: Path,
    results_json: Path,
    prompt_file: Path,
    out_json: Path,
    *,
    ignore_version: bool = True,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    with_fetch_abstract: bool = True,
) -> int:
    # 读取纯新 id（第二步产物）
    new_ids = _read_ids(new_ids_file)
    if not new_ids:
        # 写空输出
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text("[]", encoding="utf-8")
        return 0

    # 读取第一步 results.json（全集），筛出纯新条目
    items = _load_results(results_json)
    norm_new = {_normalize_id(x, ignore_version) for x in new_ids}
    selected: List[PaperMeta] = []
    for p in items:
        key = _normalize_id(p.arxiv_id, ignore_version)
        if key in norm_new:
            selected.append(p)

    if not selected:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text("[]", encoding="utf-8")
        return 0

    # 确保有摘要（没就补拉）
    if with_fetch_abstract:
        ensure_abstracts(selected, ignore_version=ignore_version, max_batch=50)

    # 准备 LLM 客户端
    cfg = LLMConfig(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY",""),
        base_url=base_url,
        model=model,
    )
    client = _client_from_env(cfg)
    system_prompt = _load_prompt_text(prompt_file)

    # 逐篇生成三字段
    enriched = []
    for p in selected:
        triplet = call_llm_for_paper(client, cfg.model, system_prompt, p)
        enriched.append({
            # 注意：这里按你最终期望的字段名输出（空格、拼写一致）
            "arXiv id": p.arxiv_id,
            "date": p.date,
            "title": p.title,
            "arXiv pdf adress": p.pdf,
            "author list": p.authors,
            "problem": triplet.get("problem",""),
            "method":  triplet.get("method",""),
            "result":  triplet.get("result",""),
            "theme": p.theme,
        })

    # 写出目标文件
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(enriched)

def main():
    ap = argparse.ArgumentParser(description="Step-3: summarize pure-new arXiv papers via DeepSeek-compatible API.")
    ap.add_argument("--new-ids", type=Path, default=Path("data/new_ids.txt"),
                    help="pure-new ids (from step-2)")
    ap.add_argument("--results", type=Path, default=Path("results.json"),
                    help="full results.json (from step-1)")
    ap.add_argument("--prompt-file", type=Path, required=True,
                    help="path to your system prompt file (you write the content)")
    ap.add_argument("--out", type=Path, default=Path(_today_filename())),
                    # 默认 DD_MM_YY_Daily_Arxiv.json
    ap.add_argument("--keep-version", action="store_true",
                    help="match IDs with version (default ignores version)")
    ap.add_argument("--api-key", type=str, default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env)")
    ap.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="OpenAI-compatible base url")
    ap.add_argument("--model", type=str, default="deepseek-chat", help="model name")
    ap.add_argument("--no-fetch-abstract", action="store_true",
                    help="do not fetch abstract if missing")

    args = ap.parse_args()

    n = summarize_pipeline(
        new_ids_file=args.new_ids,
        results_json=args.results,
        prompt_file=args.prompt_file,
        out_json=args.out,
        ignore_version=not args.keep_version,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        with_fetch_abstract=not args.no_fetch_abstract,
    )
    print(f"Summarized papers: {n}")