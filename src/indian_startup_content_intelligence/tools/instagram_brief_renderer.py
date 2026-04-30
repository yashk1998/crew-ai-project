"""Deterministic renderer: InstagramBriefBatch -> polished HTML production document.

Renders directly from the typed Pydantic batch with no LLM in the loop, so
output is byte-stable for a given input.
"""

from __future__ import annotations

import base64
import html as html_module
import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from indian_startup_content_intelligence.models import (
    InstagramBrief,
    InstagramBriefBatch,
)
from indian_startup_content_intelligence.tools.html_uploader import upload_html


_CSS = """
  :root {
    --ink: #0f172a;
    --muted: #64748b;
    --accent: #1f4e79;
    --accent-soft: #eaf1f8;
    --accent-deep: #143a5e;
    --rule: #e2e8f0;
    --bg: #ffffff;
    --bg-alt: #f8fafc;
    --tag-bg: #f1f5f9;
    --hook-bg: #fff8e6;
    --caption-bg: #f0fbf4;
    --good: #1e7c3a;
    --warn: #8a6300;
    --bad: #a31d1d;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', 'Calibri', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: var(--ink);
    background: var(--bg);
    max-width: 9in;
    margin: 0.5in auto;
    padding: 0.4in 0.7in;
  }
  header.doc {
    border-bottom: 4px solid var(--accent);
    padding-bottom: 14pt;
    margin-bottom: 24pt;
  }
  header.doc h1 {
    font-size: 26pt;
    font-weight: 800;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.4pt;
  }
  header.doc .subtitle {
    color: var(--muted);
    font-size: 10.5pt;
    margin-top: 6pt;
  }
  nav.toc {
    background: var(--accent-soft);
    border-radius: 8pt;
    padding: 16pt 22pt;
    margin-bottom: 28pt;
  }
  nav.toc h2 {
    margin: 0 0 10pt 0;
    color: var(--accent);
    font-size: 11pt;
    text-transform: uppercase;
    letter-spacing: 0.6pt;
  }
  nav.toc ol { margin: 0; padding-left: 22pt; }
  nav.toc li { margin-bottom: 5pt; }
  nav.toc a { color: var(--ink); text-decoration: none; font-weight: 600; }
  nav.toc a:hover { text-decoration: underline; }
  nav.toc .meta { color: var(--muted); font-size: 10pt; margin-left: 6pt; }
  article.brief {
    border: 1px solid var(--rule);
    border-radius: 10pt;
    padding: 22pt 28pt;
    margin-bottom: 32pt;
    background: var(--bg);
    page-break-after: always;
  }
  article.brief > h2 {
    font-size: 18pt;
    color: var(--accent);
    margin: 0 0 8pt 0;
    line-height: 1.3;
  }
  article.brief > .brief-id {
    color: var(--muted);
    font-size: 10pt;
    text-transform: uppercase;
    letter-spacing: 0.4pt;
    margin-bottom: 16pt;
  }
  .strategic-frame {
    background: var(--bg-alt);
    border-left: 4px solid var(--accent);
    border-radius: 6pt;
    padding: 16pt 20pt;
    margin: 14pt 0 22pt 0;
  }
  .strategic-frame .thesis {
    font-size: 13pt;
    font-weight: 600;
    color: var(--accent-deep);
    margin: 0 0 14pt 0;
    line-height: 1.45;
  }
  .frame-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12pt 22pt;
  }
  .frame-grid .label {
    display: block;
    color: var(--muted);
    font-size: 9pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5pt;
    margin-bottom: 3pt;
  }
  section {
    margin: 22pt 0;
    padding-top: 14pt;
    border-top: 1px solid var(--rule);
  }
  section > h3 {
    font-size: 11pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6pt;
    color: var(--accent);
    margin: 0 0 10pt 0;
  }
  p { margin: 0 0 8pt 0; }
  ul, ol { margin: 6pt 0 8pt 0; padding-left: 22pt; }
  li { margin-bottom: 5pt; }
  .pill {
    display: inline-block;
    padding: 2pt 9pt;
    border-radius: 999pt;
    font-size: 9.5pt;
    font-weight: 700;
    background: var(--accent-soft);
    color: var(--accent);
  }
  .pill.verified { background: #e6f4ea; color: var(--good); }
  .pill.evergreen { background: #fff4d6; color: var(--warn); }
  .pill.unverified { background: #fde2e2; color: var(--bad); }
  .pill.format-reel { background: #fde6f3; color: #9a2266; }
  .pill.format-carousel { background: #e0ecff; color: #1d4ed8; }
  .pill.format-story { background: #fff4e0; color: #b45309; }
  .pill.format-static { background: #ecfdf5; color: #047857; }
  .pill.goal { background: var(--accent-deep); color: #fff; }
  .specs-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6pt 18pt;
    margin: 6pt 0;
  }
  .specs-grid .k { color: var(--muted); font-size: 10pt; }
  .specs-grid .v { font-weight: 600; }
  ul.sources { padding-left: 22pt; }
  ul.sources li { margin-bottom: 8pt; }
  ul.sources a { color: var(--accent); font-weight: 600; text-decoration: none; }
  ul.sources a:hover { text-decoration: underline; }
  ul.sources .meta { color: var(--muted); font-size: 9.5pt; display: block; }
  .hooks-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10pt;
  }
  .hook-card {
    background: var(--hook-bg);
    border-left: 3px solid #d97706;
    border-radius: 4pt;
    padding: 10pt 14pt;
  }
  .hook-card .hook-label {
    background: #d97706;
    color: #fff;
    width: 22pt;
    height: 22pt;
    border-radius: 999pt;
    text-align: center;
    line-height: 22pt;
    font-weight: 800;
    display: inline-block;
    margin-right: 8pt;
  }
  .hook-card .hook-angle { color: var(--muted); font-size: 9.5pt; text-transform: uppercase; letter-spacing: 0.5pt; }
  .hook-card .hook-text {
    font-size: 12.5pt;
    font-weight: 700;
    line-height: 1.4;
    margin: 4pt 0 6pt 0;
    color: var(--ink);
  }
  .hook-card .hook-rationale { font-size: 10pt; color: var(--ink); margin: 0; font-style: italic; }
  table.shot-list {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-size: 10pt;
  }
  table.shot-list th {
    background: var(--accent-soft);
    color: var(--accent);
    text-align: left;
    padding: 7pt 9pt;
    font-weight: 700;
    border: 1px solid var(--rule);
    text-transform: uppercase;
    font-size: 9pt;
    letter-spacing: 0.4pt;
  }
  table.shot-list td {
    padding: 8pt 9pt;
    border: 1px solid var(--rule);
    vertical-align: top;
  }
  table.shot-list .seq {
    font-weight: 800;
    color: var(--accent);
    text-align: center;
    background: var(--bg-alt);
  }
  table.shot-list tr:nth-child(even) td:not(.seq) { background: #fafbfc; }
  table.shot-list .design {
    color: var(--muted);
    font-size: 9pt;
    margin-top: 4pt;
    border-top: 1px dashed var(--rule);
    padding-top: 4pt;
  }
  table.shot-list .vo {
    background: #fffaf0;
    color: #9a3412;
    font-style: italic;
  }
  .caption-card {
    background: var(--caption-bg);
    border-left: 3px solid var(--good);
    border-radius: 4pt;
    padding: 12pt 16pt;
  }
  .caption-card .full-text {
    white-space: pre-wrap;
    margin: 0;
    font-family: Georgia, 'Iowan Old Style', serif;
    font-size: 11pt;
    line-height: 1.55;
  }
  .hashtags {
    color: var(--accent);
    font-weight: 600;
    background: var(--tag-bg);
    padding: 8pt 14pt;
    border-radius: 5pt;
  }
  .cta-primary {
    background: var(--accent);
    color: #fff;
    padding: 12pt 16pt;
    border-radius: 6pt;
    margin-bottom: 8pt;
    font-weight: 600;
    font-size: 11.5pt;
  }
  .cta-comment {
    background: var(--bg-alt);
    border: 1px solid var(--rule);
    padding: 10pt 14pt;
    border-radius: 5pt;
  }
  .text-block { white-space: pre-wrap; margin: 0; line-height: 1.6; }
  footer.doc {
    border-top: 1px solid var(--rule);
    padding-top: 12pt;
    margin-top: 30pt;
    color: var(--muted);
    font-size: 9.5pt;
    text-align: center;
  }
  @media print {
    body { margin: 0.4in; padding: 0; max-width: none; }
    article.brief { page-break-after: always; box-shadow: none; }
    nav.toc { page-break-after: always; }
  }
"""


def _esc(value: str) -> str:
    return html_module.escape(str(value or ""), quote=True)


def _format_label(brief: InstagramBrief) -> str:
    return brief.format.capitalize()


def _fact_check_pill(fact_check: str) -> str:
    text = fact_check or ""
    status = text.split(" — ", 1)[0].strip() if " — " in text else text.strip()
    note = text.split(" — ", 1)[1].strip() if " — " in text else ""
    cls = status.lower().replace(" ", "")
    return (
        f'<span class="pill {_esc(cls)}">{_esc(status)}</span>'
        + (f' <span>{_esc(note)}</span>' if note else "")
    )


def _render_specs(specs_text: str) -> str:
    parts = [p.strip() for p in specs_text.split("|") if p.strip()]
    if len(parts) <= 1:
        return f'<p class="body">{_esc(specs_text)}</p>'
    rows = []
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            rows.append(f'<div class="k">{_esc(k.strip())}</div><div class="v">{_esc(v.strip())}</div>')
        else:
            rows.append(f'<div class="k">·</div><div class="v">{_esc(part)}</div>')
    return f'<div class="specs-grid">{"".join(rows)}</div>'


def _render_strategic_frame(brief: InstagramBrief) -> str:
    return f"""
    <div class="strategic-frame">
      <p class="thesis">{_esc(brief.thesis)}</p>
      <div class="frame-grid">
        <div><span class="label">Target Subaudience</span>{_esc(brief.target_subaudience)}</div>
        <div><span class="label">Why Now</span>{_esc(brief.why_now)}</div>
        <div><span class="label">Primary Goal</span><span class="pill goal">{_esc(brief.primary_goal)}</span></div>
        <div><span class="label">Format</span><span class="pill format-{_esc(brief.format)}">{_esc(_format_label(brief))}</span></div>
      </div>
    </div>
    """


def _render_sources(brief: InstagramBrief) -> str:
    items = "".join(
        f'<li><a href="{_esc(c.url)}" target="_blank" rel="noopener">{_esc(c.title)}</a>'
        f'<span class="meta">{_esc(c.source)}</span></li>'
        for c in brief.source_citations
    )
    return f'<ul class="sources">{items}</ul>'


def _render_hooks(brief: InstagramBrief) -> str:
    cards = "".join(
        f"""
        <div class="hook-card">
          <span class="hook-label">{_esc(h.label)}</span>
          <span class="hook-angle">{_esc(h.angle)}</span>
          <p class="hook-text">{_esc(h.text)}</p>
          <p class="hook-rationale">{_esc(h.why_it_works)}</p>
        </div>
        """
        for h in brief.hooks
    )
    return f'<div class="hooks-grid">{cards}</div>'


def _render_shot_list(brief: InstagramBrief) -> str:
    is_video = brief.format in ("reel", "story")
    rows = "".join(
        "<tr>"
        f"<td class='seq'>{shot.sequence_number}</td>"
        f"<td><strong>{_esc(shot.label)}</strong><br>{_esc(shot.visual_concept)}<br>"
        f"<div class='design'>{_esc(shot.design_notes)}</div></td>"
        f"<td>{_esc(shot.headline)}</td>"
        f"<td class='vo'>{_esc(shot.voiceover) or '<span style=\"color:#9ca3af\">—</span>'}</td>"
        "</tr>"
        for shot in brief.slides_or_shots
    )
    vo_header = "Voiceover" if is_video else "Voice / Tease"
    return f"""
    <table class="shot-list">
      <thead>
        <tr>
          <th style="width:32pt">#</th>
          <th>Visual / Design</th>
          <th>Headline</th>
          <th>{_esc(vo_header)}</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _render_brief(brief: InstagramBrief, idx: int) -> str:
    hashtags_html = " ".join(f"#{_esc(t.lstrip('#'))}" for t in brief.hashtags)
    return f"""
    <article class="brief" id="brief-{idx}">
      <h2>{_esc(brief.topic_line)}</h2>
      <div class="brief-id">Brief #{idx}</div>

      {_render_strategic_frame(brief)}

      <section>
        <h3>📋 Specs &amp; Fact-Check</h3>
        {_render_specs(brief.specs)}
        <p style="margin-top:8pt">{_fact_check_pill(brief.fact_check)}</p>
      </section>

      <section>
        <h3>📚 Sources Cited</h3>
        {_render_sources(brief)}
      </section>

      <section>
        <h3>🎣 Hook Options</h3>
        {_render_hooks(brief)}
      </section>

      <section>
        <h3>🎬 Slide / Shot Blueprint</h3>
        {_render_shot_list(brief)}
      </section>

      <section>
        <h3>✏️ Caption</h3>
        <div class="caption-card">
          <p class="full-text">{_esc(brief.caption)}</p>
        </div>
      </section>

      <section>
        <h3># Hashtags</h3>
        <p class="hashtags">{hashtags_html}</p>
      </section>

      <section>
        <h3>📣 CTAs</h3>
        <div class="cta-primary">{_esc(brief.primary_cta)}</div>
        <div class="cta-comment"><strong>Comment-driver:</strong> {_esc(brief.comment_cta)}</div>
      </section>

      <section>
        <h3>🎵 Audio</h3>
        <p class="text-block">{_esc(brief.audio_recommendation)}</p>
      </section>

      <section>
        <h3>📅 Distribution &amp; Engagement</h3>
        <p class="text-block">{_esc(brief.distribution_notes)}</p>
      </section>
    </article>
    """


def _render_toc(batch: InstagramBriefBatch) -> str:
    items = "".join(
        f'<li><a href="#brief-{i}">{_esc(b.topic_line)}</a>'
        f'<span class="meta"> — {_esc(_format_label(b))}'
        f' · {_esc(b.fact_check.split(" — ", 1)[0])}</span></li>'
        for i, b in enumerate(batch.briefs, start=1)
    )
    return (
        '<nav class="toc">'
        '<h2>Briefs in this document</h2>'
        f'<ol>{items}</ol>'
        '</nav>'
    )


def render_briefs_to_html(
    batch: InstagramBriefBatch,
    *,
    title: str = "Instagram Content Briefs — Indian Founder Edition",
) -> str:
    briefs_html = "".join(
        _render_brief(b, i) for i, b in enumerate(batch.briefs, start=1)
    )
    count = len(batch.briefs)
    plural = "s" if count != 1 else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <header class="doc">
    <h1>{_esc(title)}</h1>
    <div class="subtitle">{count} production-ready Instagram brief{plural} · For Indian early-stage SaaS / B2B founders · Generated by the Indian Startup Content Intelligence Crew</div>
  </header>
  {_render_toc(batch)}
  {briefs_html}
  <footer class="doc">
    Open this file in your browser, or open it directly in Microsoft Word. Every claim cites a real source URL — verify before publishing.
  </footer>
</body>
</html>
"""


# ---------- Markdown renderer (AMP-friendly, rendered inline in dashboard) ----------

def _md_escape_pipe(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _render_brief_md(brief: InstagramBrief, idx: int) -> str:
    parts: list[str] = []
    fmt_emoji = {"reel": "🎬", "carousel": "🖼️", "story": "📱", "static": "🖋️"}.get(brief.format, "📄")
    parts.append(f"## {fmt_emoji} Brief #{idx} — {brief.topic_line}")
    parts.append("")

    parts.append(f"> **{brief.thesis}**")
    parts.append("")
    parts.append(
        f"**Audience:** {brief.target_subaudience}  \n"
        f"**Why now:** {brief.why_now}  \n"
        f"**Format:** `{brief.format}`  ·  **Goal:** `{brief.primary_goal}`  ·  **Specs:** {brief.specs}  \n"
        f"**Fact-check:** {brief.fact_check}"
    )
    parts.append("")

    parts.append("### 📚 Sources")
    for c in brief.source_citations:
        parts.append(f"- [{c.title}]({c.url}) — *{c.source}*")
    parts.append("")

    parts.append("### 🎣 Hooks")
    for h in brief.hooks:
        parts.append(f"**{h.label} · {h.angle}** — {h.text}")
        parts.append(f"  _{h.why_it_works}_")
    parts.append("")

    parts.append("### 🎬 Slide / Shot Blueprint")
    parts.append("| # | Slide / Frame | Visual | Headline | Voiceover |")
    parts.append("|---|---|---|---|---|")
    for s in brief.slides_or_shots:
        parts.append(
            "| "
            + " | ".join(
                _md_escape_pipe(c)
                for c in [
                    str(s.sequence_number),
                    s.label,
                    f"{s.visual_concept} _(design: {s.design_notes})_",
                    s.headline,
                    s.voiceover or "—",
                ]
            )
            + " |"
        )
    parts.append("")

    parts.append("### ✏️ Caption")
    parts.append("> " + brief.caption.replace("\n", "  \n> "))
    parts.append("")

    parts.append("### # Hashtags")
    parts.append(" ".join(f"`#{tag.lstrip('#')}`" for tag in brief.hashtags))
    parts.append("")

    parts.append("### 📣 CTAs")
    parts.append(f"- **Primary:** {brief.primary_cta}")
    parts.append(f"- **Comment-driver:** {brief.comment_cta}")
    parts.append("")

    parts.append("### 🎵 Audio")
    parts.append(brief.audio_recommendation)
    parts.append("")

    parts.append("### 📅 Distribution & Engagement")
    parts.append(brief.distribution_notes)
    parts.append("")

    return "\n".join(parts)


def render_briefs_to_markdown(batch: InstagramBriefBatch) -> str:
    """Render the batch as rich markdown — for AMP UI inline preview."""
    out: list[str] = []
    out.append("# Instagram Content Briefs — Indian Founder Edition")
    out.append("")
    out.append(
        f"_{len(batch.briefs)} production-ready briefs · For Indian early-stage SaaS / B2B founders · Generated by the Indian Startup Content Intelligence Crew._"
    )
    out.append("")
    out.append("**Briefs in this document:**")
    for i, b in enumerate(batch.briefs, start=1):
        out.append(f"  {i}. {b.topic_line} (`{b.format}`)")
    out.append("")
    out.append("---")
    out.append("")
    for i, brief in enumerate(batch.briefs, start=1):
        out.append(_render_brief_md(brief, i))
        out.append("---")
        out.append("")
    return "\n".join(out)


def html_to_data_url(html: str) -> str:
    """Encode HTML as a base64 data URL — clickable, opens fully styled in browser."""
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded}"


# ---------- Tool wrapper ----------

class InstagramBriefRendererInput(BaseModel):
    briefs_markdown: str = Field(
        ...,
        description=(
            "The full markdown of the 3 Instagram briefs from the previous "
            "task. Pass it verbatim — the tool wraps it in a styled HTML "
            "document and uploads to a public host."
        ),
    )


class InstagramBriefRendererTool(BaseTool):
    """Wrap brief-markdown in styled HTML, upload to catbox, return shareable URL.

    Input: the brief generator's markdown output verbatim.
    Output: a banner with a clickable public URL + the original markdown.
    """

    name: str = "instagram_brief_renderer"
    description: str = (
        "Wraps the previous task's brief markdown in a professionally styled "
        "HTML document, uploads it to a public host (catbox.moe), and returns "
        "a banner with a clickable shareable URL plus the original markdown. "
        "Pass the briefs as `briefs_markdown` verbatim. Return the tool's "
        "output verbatim."
    )
    args_schema: Type[BaseModel] = InstagramBriefRendererInput

    def _run(self, briefs_markdown: str) -> str:
        # Wrap the markdown in styled HTML using the existing formatter.
        from indian_startup_content_intelligence.tools.professional_document_formatter import (
            ProfessionalDocumentFormatter,
        )
        try:
            html = ProfessionalDocumentFormatter()._run(content=briefs_markdown)
        except Exception as exc:
            html = (
                "<!DOCTYPE html><html><body>"
                f"<pre>{html_module.escape(briefs_markdown)}</pre>"
                f"<hr><pre>HTML wrap failed: {html_module.escape(str(exc))}</pre>"
                "</body></html>"
            )

        # Upload to a public file host for a real, shareable URL.
        public_url = upload_html(html, filename="indian-founder-briefs.html")

        if public_url:
            banner = (
                f"# 🔗 [📄 View the polished briefs document]({public_url})\n\n"
                f"**Direct link:** {public_url}\n\n"
                "_Opens in any browser as a fully-styled page · Imports cleanly "
                "into Microsoft Word · Share this URL with anyone, no login needed._\n\n"
                "---\n\n"
            )
        else:
            data_url = html_to_data_url(html)
            banner = (
                f"> 🔗 **[Open the fully-styled HTML]({data_url})**\n\n"
                "_(Public-host upload failed — copy the link above into a "
                "browser address bar.)_\n\n"
                "---\n\n"
            )
        return banner + briefs_markdown
