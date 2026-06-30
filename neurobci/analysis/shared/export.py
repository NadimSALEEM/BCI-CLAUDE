"""Exporters: CSV / JSON / figures / self-contained HTML (and PDF of figures).

Kept dependency-light: CSV via the stdlib, figures via the matplotlib the app
already ships. The :class:`HTMLReport` builder assembles a single portable HTML
file with figures embedded as base64 PNG -- no external asset files to lose.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ----- primitive savers -------------------------------------------------- #

def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def save_table_csv(columns: dict[str, list], path: str | Path) -> Path:
    """Write a dict-of-columns table to CSV (columns need equal length)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(columns.keys())
    n = max((len(v) for v in columns.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i in range(n):
            w.writerow([columns[h][i] if i < len(columns[h]) else "" for h in headers])
    return path


def save_figure(fig, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a matplotlib Figure; format inferred from extension (png/svg/pdf)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def figures_to_pdf(figs, path: str | Path) -> Path:
    """Bundle several matplotlib figures into one PDF (best-effort PDF report)."""
    from matplotlib.backends.backend_pdf import PdfPages
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def _fig_to_b64_png(fig, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----- HTML report builder ----------------------------------------------- #

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;
 color:#1c2530;background:#fafbfc;max-width:1000px;}
h1{font-size:22px;} h2{font-size:17px;margin-top:26px;border-bottom:1px solid #e2e6ea;
 padding-bottom:4px;} table{border-collapse:collapse;margin:8px 0;font-size:13px;}
th,td{border:1px solid #d6dbe0;padding:4px 9px;text-align:left;}
th{background:#eef2f5;} .kv td:first-child{color:#5a6672;width:230px;}
figure{margin:14px 0;} figcaption{font-size:12px;color:#5a6672;}
.warn{color:#9a6b00;} .err{color:#b00020;} .info{color:#3a6ea5;}
.muted{color:#7a8590;font-size:12px;} code{background:#eef2f5;padding:1px 4px;border-radius:3px;}
"""


class HTMLReport:
    """Incrementally build a single self-contained HTML report."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._parts: list[str] = []

    def add_heading(self, text: str, level: int = 2) -> "HTMLReport":
        self._parts.append(f"<h{level}>{html.escape(text)}</h{level}>")
        return self

    def add_paragraph(self, text: str, css: str = "") -> "HTMLReport":
        cls = f' class="{css}"' if css else ""
        self._parts.append(f"<p{cls}>{html.escape(text)}</p>")
        return self

    def add_keyvalue(self, mapping: dict) -> "HTMLReport":
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in mapping.items())
        self._parts.append(f'<table class="kv">{rows}</table>')
        return self

    def add_table(self, headers: list[str], rows: list[list]) -> "HTMLReport":
        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
            for r in rows)
        self._parts.append(f"<table><tr>{head}</tr>{body}</table>")
        return self

    def add_diagnostics(self, diagnostics) -> "HTMLReport":
        if not diagnostics:
            return self
        items = []
        for d in diagnostics:
            level = getattr(d, "level", "info")
            msg = getattr(d, "message", str(d))
            cls = {"warning": "warn", "error": "err"}.get(level, "info")
            items.append(f'<li class="{cls}">{html.escape(msg)}</li>')
        self._parts.append("<ul>" + "".join(items) + "</ul>")
        return self

    def add_figure(self, fig, caption: str = "") -> "HTMLReport":
        try:
            b64 = _fig_to_b64_png(fig)
            cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
            self._parts.append(
                f'<figure><img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%"/>{cap}</figure>')
        except Exception:  # noqa: BLE001 - a bad figure must not sink the report
            logger.exception("Could not embed figure in report.")
        return self

    def to_html(self) -> str:
        body = "\n".join(self._parts)
        return (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{html.escape(self.title)}</title><style>{_CSS}</style>"
                f"</head><body><h1>{html.escape(self.title)}</h1>{body}"
                f"<p class='muted'>Generated by NeuroBCI analysis suite.</p>"
                f"</body></html>")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path
