from pydantic import BaseModel, Field
from typing import List, Optional

class ThemeSpec(BaseModel):
    name: str
    priority: int = Field(ge=1, le=5)
    keywords: List[str]
    arxiv_categories: List[str]

class ThemeConfig(BaseModel):
    themes: List[ThemeSpec]

class PaperItem(BaseModel):
    # —— 你最初要求的字段（保持不变）——
    arXiv_id: str
    date: str
    title: str
    arXiv_pdf_adress: str
    author_list: List[str]
    porblem: str = ""
    method: str = ""
    result: str = ""
    theme: str

    # —— 可选扩展字段（默认 None，不会出现在 JSON：exclude_none=True）——
    abstract: Optional[str] = None
    score: Optional[float] = None
    primary_category: Optional[str] = None
    categories: Optional[List[str]] = None