from typing import List
from dataclasses import dataclass
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ThemeSpec

@dataclass
class ScoreResult:
    theme_name: str
    score: float

def _build_theme_texts(themes: List[ThemeSpec]) -> List[str]:
    return [" ".join([t.name] + t.keywords) for t in themes]

class PaperScorer:
    """
    评分 = priority * (0.6 * 关键词模糊 + 0.4 * TF-IDF 余弦)
    """
    def __init__(self, themes: List[ThemeSpec]):
        self.themes = themes
        self.theme_texts = _build_theme_texts(themes)
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=20000)
        self._theme_vecs = self._vectorizer.fit_transform(self.theme_texts)

    def score(self, title: str, summary: str) -> ScoreResult:
        title = title or ""
        summary = summary or ""
        text = f"{title} {summary}".strip()
        if not text:
            return ScoreResult(theme_name=self.themes[0].name, score=0.0)

        # 关键词（标题权重更高）
        kw_scores = []
        for t in self.themes:
            best = 0.0
            for kw in t.keywords:
                s_title = fuzz.token_set_ratio(kw, title) / 100.0
                s_sum = fuzz.token_set_ratio(kw, summary) / 100.0
                best = max(best, 0.7 * s_title + 0.3 * s_sum)
            kw_scores.append(best)

        # TF-IDF
        doc_vec = self._vectorizer.transform([text])
        sem_scores = cosine_similarity(doc_vec, self._theme_vecs)[0]

        best_idx, best_total = 0, -1e9
        for i, t in enumerate(self.themes):
            total = t.priority * (0.6 * kw_scores[i] + 0.4 * sem_scores[i])
            if total > best_total:
                best_total, best_idx = total, i

        return ScoreResult(theme_name=self.themes[best_idx].name, score=best_total)