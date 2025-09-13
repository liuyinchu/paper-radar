from typing import List, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from dateutil import parser as dtparser
import warnings
import arxiv

@dataclass
class RawArxiv:
    id: str
    title: str
    summary: str
    published: datetime
    updated: datetime
    authors: List[str]
    pdf_url: str
    primary_category: str
    categories: List[str]

def _entry_to_raw(e: arxiv.Result) -> RawArxiv:
    return RawArxiv(
        id=e.entry_id.split("/")[-1],
        title=e.title.strip(),
        summary=e.summary.strip(),
        published=e.published if isinstance(e.published, datetime) else dtparser.parse(str(e.published)),
        updated=e.updated   if isinstance(e.updated, datetime)   else dtparser.parse(str(e.updated)),
        authors=[a.name for a in e.authors],
        pdf_url=e.pdf_url,
        primary_category=e.primary_category,
        categories=list(e.categories) if e.categories else [],
    )

def build_category_query(categories: List[str]) -> str:
    cats = sorted(set(categories))
    if not cats:
        return ""
    ors = " OR ".join([f"cat:{c}" for c in cats])
    return f"({ors})"

def fetch_recent_pool(
    categories: List[str],
    max_results: int = 400,
    page_size: int = 100,
    delay_seconds: float = 3.1,
    num_retries: int = 3,
) -> List[RawArxiv]:
    """
    单连接、分页延时，且对偶发“空页”容错。
    为减少“第二页空”的概率，将 page_size 动态收敛到 ≤ max_results（通常只打一页）。
    """
    query = build_category_query(categories)
    # 关键：把单页大小设为不超过 max_results，且不超过 200（保守上限）
    effective_page_size = min(max_results, 200, page_size if page_size > 0 else 100)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(
        page_size=effective_page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )

    seen: Set[str] = set()
    out: List[RawArxiv] = []
    try:
        for e in client.results(search):
            r = _entry_to_raw(e)
            if r.id not in seen:
                seen.add(r.id)
                out.append(r)
    except arxiv.UnexpectedEmptyPageError as ex:
        # 不中断，返回目前已收集到的结果；给出一次温和警告
        warnings.warn(f"arXiv returned an empty page (tolerated): {ex}; returning {len(out)} collected items.")
    except Exception as ex:
        # 其他异常也别让流程爆掉，返回已收集的
        warnings.warn(f"arXiv fetch encountered {type(ex).__name__}: {ex}; returning {len(out)} collected items.")

    return out