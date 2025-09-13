import json
from pathlib import Path
from typing import List
import argparse
import yaml
from datetime import datetime, timezone, timedelta

from .models import ThemeConfig, PaperItem
from .scorer import PaperScorer
from .arxiv_client import fetch_recent_pool

# -------- helpers & cache --------

CACHE_DIR = Path(".paperscout_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _load_themes(path: Path) -> ThemeConfig:
    return ThemeConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))

def _collect_all_categories(cfg: ThemeConfig) -> List[str]:
    cats = []
    for t in cfg.themes:
        cats.extend(t.arxiv_categories)
    return sorted(set(cats))

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _cache_key(n: int) -> Path:
    today = _utc_now().strftime("%Y-%m-%d")
    return CACHE_DIR / f"daily_{today}_n{n}.json"

def _read_cache(cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _write_cache(cache_path: Path, items: List[dict]) -> None:
    cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def _partition_by_window(raws, hrs: float):
    """按 submittedDate/published 的‘近 hrs 小时’划分。"""
    now = _utc_now()
    cutoff = now - timedelta(hours=hrs)
    in_w, out_w = [], []
    for r in raws:
        if r.published >= cutoff:
            in_w.append(r)
        else:
            out_w.append(r)
    return in_w, out_w

def _same_utc_day(a: datetime, b: datetime) -> bool:
    start = datetime(b.year, b.month, b.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start <= a < end

# -------- core --------

def run(
    themes_file: Path,
    out_json: Path,
    out_ids: Path,
    top_n: int = 100,
    pool_size: int = 150,
    force: bool = False,
    strict_today: bool = False,
    min_score: float = 0.0,
    with_abstract: bool = False,
    with_score: bool = False,
    with_categories: bool = False,
):
    cfg = _load_themes(themes_file)
    all_categories = _collect_all_categories(cfg)

    # 日级缓存（仅在严格相同 top_n 的情况下复用）
    cache_path = _cache_key(top_n)
    if not force and not (strict_today or min_score > 0 or with_abstract or with_score or with_categories):
        cached = _read_cache(cache_path)
        if cached is not None:
            out_json.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            ids = [it["arXiv_id"] for it in cached]
            out_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")
            print(f"[cache] Reused cached results for today -> {out_json}")
            print(f"[cache] Reused cached ids -> {out_ids}")
            return

    scorer = PaperScorer(cfg.themes)

    # 单页优先（避免“空页”），遵守延时；按 submittedDate 降序
    raws = fetch_recent_pool(
        all_categories,
        max_results=pool_size,
        page_size=min(pool_size, 200),
        delay_seconds=3.1,
        num_retries=3,
    )

    # 选择候选
    if strict_today:
        today = _utc_now()
        candidates = [r for r in raws if _same_utc_day(r.published, today)]
    else:
        # 滚动时间窗：36h -> 72h -> 用余量补齐
        w36, remain = _partition_by_window(raws, 36.0)
        if len(w36) < top_n:
            w72_more, remain2 = _partition_by_window(remain, 72.0)
            candidates = w36 + w72_more
            tail = remain2
        else:
            candidates = w36
            tail = remain
        if len(candidates) < top_n:
            candidates += tail

    # 评分 & 过滤 & 排序
    scored = []
    for r in candidates:
        s = scorer.score(r.title, r.summary)
        if s.score >= min_score:
            scored.append((s.score, s.theme_name, r))
    scored.sort(key=lambda x: (x[0], x[2].published), reverse=True)
    picked = scored[:min(top_n, len(scored))]

    items: List[PaperItem] = []
    ids: List[str] = []
    for s_score, theme_name, r in picked:
        item = PaperItem(
            arXiv_id=r.id,
            date=r.published.strftime("%Y-%m-%d"),
            title=r.title,
            arXiv_pdf_adress=r.pdf_url,
            author_list=r.authors,
            theme=theme_name,
        )
        if with_abstract:
            item.abstract = r.summary
        if with_score:
            item.score = float(s_score)
        if with_categories:
            item.primary_category = r.primary_category
            item.categories = r.categories
        items.append(item)
        ids.append(r.id)

    payload = [i.model_dump(exclude_none=True) for i in items]
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")

    # 仅在“默认模式”下写入缓存（确保缓存内容结构稳定）
    if not (strict_today or min_score > 0 or with_abstract or with_score or with_categories):
        _write_cache(cache_path, payload)

    print(f"Saved {len(items)} items -> {out_json}")
    print(f"Saved {len(ids)} arXiv ids -> {out_ids}")

def main():
    p = argparse.ArgumentParser(description="Fetch daily arXiv papers (ToU-compliant).")
    p.add_argument("--themes", type=Path, default=Path(__file__).parent / "themes.yaml")
    p.add_argument("--out-json", type=Path, default=Path("results.json"))
    p.add_argument("--out-ids", type=Path, default=Path("ids.txt"))
    p.add_argument("--n", type=int, default=100, help="how many papers to keep")
    p.add_argument("--pool", type=int, default=150, help="pool size before scoring (<=200 recommended)")
    p.add_argument("--force", action="store_true", help="ignore daily cache and force fresh fetch")

    # 新增控制项
    p.add_argument("--strict-today", action="store_true", help="only keep UTC 'today' submissions (published)")
    p.add_argument("--min-score", type=float, default=0.0, help="drop papers with score < MIN_SCORE")
    p.add_argument("--with-abstract", action="store_true", help="include abstract in JSON")
    p.add_argument("--with-score", action="store_true", help="include score in JSON")
    p.add_argument("--with-categories", action="store_true", help="include primary_category/categories in JSON")

    args = p.parse_args()
    run(
        args.themes, args.out_json, args.out_ids,
        top_n=args.n, pool_size=args.pool, force=args.force,
        strict_today=args.strict_today,
        min_score=args.min_score,
        with_abstract=args.with_abstract,
        with_score=args.with_score,
        with_categories=args.with_categories,
    )