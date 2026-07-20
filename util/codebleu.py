from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable, Sequence


PYTHON_KEYWORDS = set(keyword.kwlist)
IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.COMMENT,
}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def normalize_code(code: str) -> str:
    return (code or "").replace("\r\n", "\n").strip()


def detect_codebleu_lang(samples: Sequence[dict]) -> str | None:
    success = 0
    total = 0
    for sample in samples[:10]:
        text = normalize_code(sample.get("answer", ""))
        if not text:
            continue
        total += 1
        try:
            ast.parse(text)
            success += 1
        except SyntaxError:
            continue
    if total and success >= max(1, total // 2):
        return "python"
    return None


def _fallback_tokenize(code: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|:=|->|[-+*/%<>{}()[\].,:=]", code)


def tokenize_python_code(code: str) -> list[str]:
    code = normalize_code(code)
    if not code:
        return []

    tokens: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in IGNORED_TOKEN_TYPES:
                continue
            tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return _fallback_tokenize(code)
    return tokens


def _join_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _token_weight(token: str) -> float:
    return 1.0 if token in PYTHON_KEYWORDS else 0.2


def _ngram_weight(ngram: Sequence[str]) -> float:
    return _mean([_token_weight(tok) for tok in ngram])


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_python_ngram_match(hypothesis: str, reference: str) -> float:
    from sacrebleu.metrics import BLEU

    hyp_tokens = tokenize_python_code(hypothesis)
    ref_tokens = tokenize_python_code(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    bleu = BLEU(tokenize="none", effective_order=True)
    return float(bleu.sentence_score(_join_tokens(hyp_tokens), [_join_tokens(ref_tokens)]).score)


def compute_python_weighted_ngram_match(hypothesis: str, reference: str, max_order: int = 4) -> float:
    hyp_tokens = tokenize_python_code(hypothesis)
    ref_tokens = tokenize_python_code(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    scores = []
    for order in range(1, max_order + 1):
        hyp_counts = Counter(_ngrams(hyp_tokens, order))
        ref_counts = Counter(_ngrams(ref_tokens, order))
        if not hyp_counts or not ref_counts:
            scores.append(0.0)
            continue

        matched_weight = 0.0
        hyp_total_weight = 0.0
        ref_total_weight = 0.0

        for ngram, count in hyp_counts.items():
            weight = _ngram_weight(ngram)
            hyp_total_weight += count * weight
            matched_weight += min(count, ref_counts.get(ngram, 0)) * weight

        for ngram, count in ref_counts.items():
            ref_total_weight += count * _ngram_weight(ngram)

        precision = matched_weight / hyp_total_weight if hyp_total_weight else 0.0
        recall = matched_weight / ref_total_weight if ref_total_weight else 0.0
        scores.append(_f1(precision, recall))

    return _mean(scores) * 100.0


def _ast_node_sequence(code: str) -> list[str]:
    tree = ast.parse(normalize_code(code))
    nodes: list[str] = []

    class Visitor(ast.NodeVisitor):
        def generic_visit(self, node):
            nodes.append(type(node).__name__)
            super().generic_visit(node)

    Visitor().visit(tree)
    return nodes


def compute_python_syntax_match(hypothesis: str, reference: str) -> float:
    try:
        hyp_nodes = _ast_node_sequence(hypothesis)
        ref_nodes = _ast_node_sequence(reference)
    except SyntaxError:
        return 0.0

    if not hyp_nodes or not ref_nodes:
        return 0.0
    return float(SequenceMatcher(a=ref_nodes, b=hyp_nodes).ratio() * 100.0)


def _extract_names(node: ast.AST | None) -> set[str]:
    names: set[str] = set()
    if node is None:
        return names
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names


def _extract_targets(node: ast.AST | None) -> set[str]:
    targets: set[str] = set()
    if node is None:
        return targets
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            targets.add(child.id)
    return targets


class _DataflowVisitor(ast.NodeVisitor):
    def __init__(self):
        self.edges: set[tuple[str, str, str]] = set()

    def _add_flow(self, sources: set[str], targets: set[str], edge_type: str):
        for src in sources:
            for tgt in targets:
                self.edges.add((src, tgt, edge_type))

    def visit_Assign(self, node: ast.Assign):
        self._add_flow(_extract_names(node.value), _extract_targets(node), "assign")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._add_flow(_extract_names(node.value), _extract_targets(node.target), "ann_assign")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        sources = _extract_names(node.value) | _extract_names(node.target)
        self._add_flow(sources, _extract_targets(node.target), "aug_assign")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._add_flow(_extract_names(node.iter), _extract_targets(node.target), "for")
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self._add_flow(_extract_names(item.context_expr), _extract_targets(item.optional_vars), "with")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        for src in _extract_names(node.value):
            self.edges.add((src, "<return>", "return"))
        self.generic_visit(node)


def _extract_dataflow_edges(code: str) -> set[tuple[str, str, str]]:
    tree = ast.parse(normalize_code(code))
    visitor = _DataflowVisitor()
    visitor.visit(tree)
    return visitor.edges


def compute_python_dataflow_match(hypothesis: str, reference: str) -> float:
    try:
        hyp_edges = _extract_dataflow_edges(hypothesis)
        ref_edges = _extract_dataflow_edges(reference)
    except SyntaxError:
        return 0.0

    if not hyp_edges and not ref_edges:
        hyp_tokens = set(tokenize_python_code(hypothesis))
        ref_tokens = set(tokenize_python_code(reference))
        if not hyp_tokens or not ref_tokens:
            return 0.0
        overlap = len(hyp_tokens & ref_tokens)
        precision = overlap / len(hyp_tokens)
        recall = overlap / len(ref_tokens)
        return _f1(precision, recall) * 100.0

    overlap = len(hyp_edges & ref_edges)
    precision = overlap / len(hyp_edges) if hyp_edges else 0.0
    recall = overlap / len(ref_edges) if ref_edges else 0.0
    return _f1(precision, recall) * 100.0


def compute_codebleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    lang: str = "python",
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> dict[str, float | str]:
    if lang != "python":
        raise NotImplementedError(f"Unsupported CodeBLEU language: {lang}")
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    ngram_scores = []
    weighted_ngram_scores = []
    syntax_scores = []
    dataflow_scores = []

    for hyp, ref in zip(hypotheses, references):
        ngram_scores.append(compute_python_ngram_match(hyp, ref))
        weighted_ngram_scores.append(compute_python_weighted_ngram_match(hyp, ref))
        syntax_scores.append(compute_python_syntax_match(hyp, ref))
        dataflow_scores.append(compute_python_dataflow_match(hyp, ref))

    ngram = _mean(ngram_scores)
    weighted_ngram = _mean(weighted_ngram_scores)
    syntax = _mean(syntax_scores)
    dataflow = _mean(dataflow_scores)
    score = (
        weights[0] * ngram
        + weights[1] * weighted_ngram
        + weights[2] * syntax
        + weights[3] * dataflow
    )

    return {
        "lang": lang,
        "score": float(score),
        "ngram_match_score": float(ngram),
        "weighted_ngram_match_score": float(weighted_ngram),
        "syntax_match_score": float(syntax),
        "dataflow_match_score": float(dataflow),
    }
