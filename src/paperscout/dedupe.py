from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Set, List
import json
from datetime import datetime, timezone

def _read_ids(path: Path) -> List[str]:
    """
    读取一个 id 列表文件：一行一个 id；支持空行/注释行(#...)；自动去重。
    """
    ids = []
    seen = set()
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            ids.append(s)
    return ids

def _strip_version(arxiv_id: str) -> str:
    """
    去掉 arXiv 版本号（例如 2509.01234v2 -> 2509.01234）。
    若你想按“精确版本”对齐，可不使用此函数（通过命令行开关控制）。
    """
    # arXiv id 可能是 old-style 或 new-style，这里只简单处理后缀 vN。
    # 如果没有 vN 后缀，则原样返回。
    s = arxiv_id.strip()
    # 仅截掉末尾的 v数字
    i = s.rfind("v")
    if i != -1 and i < len(s) - 1 and s[i+1:].isdigit():
        return s[:i]
    return s

def _normalize_ids(ids: Iterable[str], ignore_version: bool) -> Set[str]:
    if ignore_version:
        return {_strip_version(x) for x in ids}
    return set(ids)

def dedupe_ids(
    current_ids_file: Path,
    total_ids_file: Path,
    out_new_ids_file: Path,
    *,
    ignore_version: bool = True,
    filter_results_json: Path | None = None,
    in_results_json: Path | None = None,
) -> int:
    """
    与历史总表对比，输出本次“纯新”id。
    可选：基于纯新 id 过滤 results.json，输出到 filter_results_json。
    返回写出的“纯新”id 数量。
    """
    current_ids = _read_ids(current_ids_file)
    total_ids   = _read_ids(total_ids_file)

    cur_norm = _normalize_ids(current_ids, ignore_version=ignore_version)
    tot_norm = _normalize_ids(total_ids,   ignore_version=ignore_version)

    # 找出“纯新”（按规范化口径）
    new_norm = cur_norm - tot_norm

    # 还原到原始 current_ids 顺序与原样（保持用户可读性）
    new_ids_in_order: List[str] = []
    norm_lookup = {_strip_version(x) if ignore_version else x: x for x in current_ids}
    for nid in current_ids:
        key = _strip_version(nid) if ignore_version else nid
        if key in new_norm and nid not in new_ids_in_order:
            new_ids_in_order.append(nid)

    # 写出“纯新 id 表”
    header = f"# new ids generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} (ignore_version={ignore_version})"
    out_new_ids_file.parent.mkdir(parents=True, exist_ok=True)
    out_new_ids_file.write_text(
        header + "\n" + "\n".join(new_ids_in_order) + ("\n" if new_ids_in_order else ""),
        encoding="utf-8",
    )

    # 可选：过滤 results.json（只保留纯新 id 的条目）
    if filter_results_json:
        if not in_results_json:
            # 默认尝试当前目录的 results.json
            in_results_json = Path("results.json")
        if in_results_json.exists():
            try:
                data = json.loads(in_results_json.read_text(encoding="utf-8"))
                # 支持两种对齐方式：按版本或忽略版本
                keep_keys = {(_strip_version(x) if ignore_version else x) for x in new_ids_in_order}
                filtered = []
                for item in data:
                    pid = item.get("arXiv_id") or item.get("arXiv id")  # 兼容可能的字段名
                    if not pid:
                        continue
                    key = _strip_version(pid) if ignore_version else pid
                    if key in keep_keys:
                        filtered.append(item)
                filter_results_json.parent.mkdir(parents=True, exist_ok=True)
                filter_results_json.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as ex:
                # 不因过滤失败而中断主流程
                print(f"[warn] failed to filter results.json: {type(ex).__name__}: {ex}")

    return len(new_ids_in_order)

def main():
    ap = argparse.ArgumentParser(
        description="Check step: compute 'pure new' arXiv IDs by diffing current run vs. historical total."
    )
    ap.add_argument("--current-ids", type=Path, default=Path("ids.txt"),
                    help="path to current run ids.txt (one id per line)")
    ap.add_argument("--total-ids", type=Path, default=Path("data/total_ids.txt"),
                    help="path to historical total ids file")
    ap.add_argument("--out-new-ids", type=Path, default=Path("data/new_ids.txt"),
                    help="path to write pure-new ids")
    ap.add_argument("--keep-version", action="store_true",
                    help="compare WITH version suffix (default is ignoring version vN)")
    ap.add_argument("--filter-json", type=Path, default=None,
                    help="optional: output filtered results.json that only contains pure-new items")
    ap.add_argument("--in-results", type=Path, default=None,
                    help="optional: path to the input results.json (defaults to ./results.json)")
    ap.add_argument("--update-total", action="store_true",
                    help="append pure-new ids to the total ids file (idempotent)")

    args = ap.parse_args()

    n = dedupe_ids(
        current_ids_file=args.current_ids,
        total_ids_file=args.total_ids,
        out_new_ids_file=args.out_new_ids,
        ignore_version=not args.keep_version,
        filter_results_json=args.filter_json,
        in_results_json=args.in_results,
    )

    # 可选：把纯新 id 追加入总表（去重 + 保序）
    if args.update_total and n > 0:
        existed = _read_ids(args.total_ids)
        existed_set = set(existed)
        new_ids = _read_ids(args.out_new_ids)
        merged = existed[:]
        for x in new_ids:
            if x not in existed_set:
                existed_set.add(x)
                merged.append(x)
        args.total_ids.parent.mkdir(parents=True, exist_ok=True)
        args.total_ids.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    print(f"Pure-new count: {n}")