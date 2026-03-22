"""
ATS Resume Optimizer
====================
ATS-optimized resumes tailored to job postings.
Llama 3.1-70B (HF Inference) + Sentence-Transformers hybrid ML scoring.

Author: Daniel Regalado
"""

import gradio as gr
import requests
import json
import re
import os
import tempfile
import traceback
import numpy as np

from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


# ─── Config ──────────────────────────────────────────────────────────────────
LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"

hf_token = os.environ.get("HF_TOKEN", "")
llm_client = InferenceClient(model=LLM_MODEL, token=hf_token)
embed_model = SentenceTransformer(EMBED_MODEL)


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def scrape_job_posting(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        for sel in ["div.job-description", "div.jobsearch-jobDescriptionText", "div.description",
                     "div.show-more-less-html__markup", "div.description__text", "article", "main"]:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return el.get_text(separator="\n", strip=True)[:8000]
        body = soup.find("body")
        return body.get_text(separator="\n", strip=True)[:8000] if body else ""
    except Exception as e:
        return f"ERROR: {e}"


def extract_resume_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def compute_embedding_scores(resume_text: str, job_description: str) -> dict:
    r_emb = embed_model.encode(resume_text, convert_to_tensor=True)
    j_emb = embed_model.encode(job_description, convert_to_tensor=True)
    overall = float(util.cos_sim(r_emb, j_emb)[0][0])

    def chunk(text, size=200):
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size) if len(words[i:i+size]) > 20]

    rc, jc = chunk(resume_text), chunk(job_description, 100)
    if not rc or not jc:
        return {"overall_similarity": round(overall * 100, 1), "section_scores": [], "weak_coverage": []}

    re_ = embed_model.encode(rc, convert_to_tensor=True)
    je_ = embed_model.encode(jc, convert_to_tensor=True)
    sim = util.cos_sim(re_, je_)

    sections = sorted([{"preview": c[:100], "sim": round(float(sim[i].max()) * 100, 1)} for i, c in enumerate(rc)], key=lambda x: -x["sim"])[:4]
    weak = sorted([{"preview": c[:100], "sim": round(float(sim[:, j].max()) * 100, 1)} for j, c in enumerate(jc)], key=lambda x: x["sim"])
    weak = [w for w in weak if w["sim"] < 50][:3]

    return {"overall_similarity": round(overall * 100, 1), "section_scores": sections, "weak_coverage": weak}


# ═════════════════════════════════════════════════════════════════════════════
#  LLM — chat_completion
# ═════════════════════════════════════════════════════════════════════════════

def call_llm(system_msg: str, user_msg: str, max_tokens: int = 3000) -> str:
    resp = llm_client.chat_completion(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_tokens=max_tokens, temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def analyze_keywords(resume_text: str, job_description: str) -> dict:
    system_msg = """You are an expert ATS analyst and career coach. You evaluate resumes against job descriptions with nuance.

IMPORTANT SCORING RULES:
- Look for BOTH exact matches AND transferable/related skills
- "Python" matches "programming", "data analysis" partially matches "data-driven content optimization"
- Consider synonyms: "team lead" = "management", "ML" = "machine learning" = "AI"
- Give PARTIAL CREDIT for related experience (e.g., "marketing analytics" partially matches "SEO")
- A candidate with 60% of keywords but strong related experience should score 55-65%, not 20%
- Be realistic but generous — most real candidates score 40-70% on good-fit roles
- Only score below 30% if the resume is truly unrelated to the job

Respond with ONLY valid JSON, no markdown backticks."""

    user_msg = f"""Analyze this resume against the job description.

For each keyword:
- "FOUND" = exact match or very close synonym exists in resume
- "PARTIAL" = related/transferable skill exists (give 0.5 credit)
- "MISSING" = no relevant experience

Calculate match_score as: (FOUND_count + 0.5 * PARTIAL_count) / total_keywords * 100

JSON format:
{{"job_title": "...", "company": "...", "ats_keywords": [{{"keyword": "...", "status": "FOUND|PARTIAL|MISSING", "evidence": "brief quote from resume or null"}}], "match_score": 55, "critical_gaps": ["..."], "recommendation_reason": "2-3 sentence assessment", "tailoring_suggestions": ["actionable suggestion 1", "..."], "strengths": ["..."]}}

RESUME:
{resume_text[:3500]}

JOB DESCRIPTION:
{job_description[:3500]}"""

    raw = call_llm(system_msg, user_msg, max_tokens=2500)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON in response: {raw[:300]}")
    raw = raw[start:]
    depth, end = 0, 0
    for i, ch in enumerate(raw):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i + 1; break
    if end: raw = raw[:end]
    return json.loads(raw)


def generate_optimized_resume(resume_text: str, job_description: str, analysis: dict) -> str:
    kw_found = [k["keyword"] for k in analysis.get("ats_keywords", []) if k["status"] in ("FOUND", "PARTIAL")]
    kw_missing = [k["keyword"] for k in analysis.get("ats_keywords", []) if k["status"] == "MISSING"]
    suggestions = analysis.get("tailoring_suggestions", [])

    system_msg = """You are an expert resume writer. Rewrite the resume to maximize ATS match for the target job.

RULES:
- NEVER invent experience, skills, or achievements not in the original
- Reorganize sections to highlight relevant experience first
- Rephrase bullets to naturally include target keywords where truthful
- Mirror the job description's terminology
- For missing keywords: if the candidate has RELATED experience, reframe it to be closer
- Keep 1-2 pages, clean Markdown: # Name, ## Sections, ### Roles, - bullets
- Add a "Skills" section near the top with relevant keywords the candidate actually has"""

    user_msg = f"""TARGET: {analysis.get('job_title', 'N/A')} at {analysis.get('company', 'N/A')}
MATCHED KEYWORDS: {json.dumps(kw_found)}
MISSING (reframe if possible): {json.dumps(kw_missing)}
SUGGESTIONS: {json.dumps(suggestions)}

ORIGINAL RESUME:
{resume_text[:4500]}

JOB DESCRIPTION:
{job_description[:2000]}

Write the optimized resume in Markdown now."""

    return call_llm(system_msg, user_msg, max_tokens=3000)


# ═════════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def markdown_to_pdf(md: str, path: str):
    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.6*inch, rightMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("RName", parent=styles["Title"], fontSize=18, leading=22, textColor=HexColor("#1a1a2e"), spaceAfter=4, alignment=TA_CENTER))
    styles.add(ParagraphStyle("RContact", parent=styles["Normal"], fontSize=9, leading=12, textColor=HexColor("#555"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle("RSec", parent=styles["Heading2"], fontSize=12, leading=15, textColor=HexColor("#1a1a2e"), spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle("RBullet", parent=styles["Normal"], fontSize=10, leading=13, leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=1, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle("RSub", parent=styles["Normal"], fontSize=10, leading=13, textColor=HexColor("#333"), spaceBefore=6, spaceAfter=2))
    styles.add(ParagraphStyle("RBody", parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_JUSTIFY))

    def clean(t):
        t = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.*?)\*", r"<i>\1</i>", t)
        return t

    story = []
    for line in md.split("\n"):
        s = line.strip()
        if not s: story.append(Spacer(1, 4))
        elif s.startswith("# ") and not s.startswith("## "): story.append(Paragraph(clean(s[2:]), styles["RName"]))
        elif s.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#1a1a2e"), spaceAfter=2, spaceBefore=6))
            story.append(Paragraph(clean(s[3:].upper()), styles["RSec"]))
        elif s.startswith("### "): story.append(Paragraph(clean(s[4:]), styles["RSub"]))
        elif s.startswith("- ") or s.startswith("* "): story.append(Paragraph(clean(s[2:]), styles["RBullet"], bulletText="•"))
        elif "|" in s or "@" in s: story.append(Paragraph(clean(s), styles["RContact"]))
        else: story.append(Paragraph(clean(s), styles["RBody"]))
    doc.build(story)


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT FORMATTER
# ═════════════════════════════════════════════════════════════════════════════

def format_report(analysis: dict, embed: dict) -> str:
    kw_score = analysis.get("match_score", 0)
    sem_score = embed.get("overall_similarity", 0)
    hybrid = round(0.6 * kw_score + 0.4 * sem_score, 1)

    if hybrid >= 70: color, label = "#0a8a2e", "Strong Match"
    elif hybrid >= 45: color, label = "#b08600", "Moderate Match"
    else: color, label = "#cc3333", "Low Match"

    r = f"""## {analysis.get('job_title', 'Position')} at {analysis.get('company', 'Company')}

### Match Score: {hybrid}% — {label}

| Component | Score | Weight |
|---|---|---|
| ATS Keywords | {kw_score}% | 60% |
| Semantic Similarity | {sem_score}% | 40% |
| **Combined** | **{hybrid}%** | |

{analysis.get('recommendation_reason', '')}

---

### ATS Keywords

"""
    # Use a list format instead of table to avoid column-width issues
    found_kws = [kw for kw in analysis.get("ats_keywords", []) if kw.get("status") == "FOUND"]
    partial_kws = [kw for kw in analysis.get("ats_keywords", []) if kw.get("status") == "PARTIAL"]
    missing_kws = [kw for kw in analysis.get("ats_keywords", []) if kw.get("status") == "MISSING"]

    if found_kws:
        r += "**Found:**\n"
        for kw in found_kws:
            ev = kw.get("evidence") or kw.get("resume_evidence") or ""
            ev = f" — *{ev[:55]}*" if ev and ev != "null" else ""
            r += f"- {kw['keyword']}{ev}\n"
        r += "\n"

    if partial_kws:
        r += "**Partial Match:**\n"
        for kw in partial_kws:
            ev = kw.get("evidence") or kw.get("resume_evidence") or ""
            ev = f" — *{ev[:55]}*" if ev and ev != "null" else ""
            r += f"- {kw['keyword']}{ev}\n"
        r += "\n"

    if missing_kws:
        r += "**Missing:**\n"
        for kw in missing_kws:
            r += f"- {kw['keyword']}\n"
        r += "\n"

    if analysis.get("strengths"):
        r += "\n### Strengths\n"
        for s in analysis["strengths"]:
            r += f"- {s}\n"

    if analysis.get("critical_gaps"):
        r += "\n### Gaps to Address\n"
        for g in analysis["critical_gaps"]:
            r += f"- {g}\n"

    if embed.get("section_scores"):
        r += "\n### Semantic Analysis — Best Resume Sections\n"
        for sec in embed["section_scores"]:
            r += f"- **{sec['sim']}%** — *{sec['preview'][:70]}...*\n"

    if embed.get("weak_coverage"):
        r += "\n### Weak Coverage Areas\n"
        for w in embed["weak_coverage"]:
            r += f"- **{w['sim']}%** — *{w['preview'][:80]}...*\n"

    if analysis.get("tailoring_suggestions"):
        r += "\n### Tailoring Strategy\n"
        for t in analysis["tailoring_suggestions"]:
            r += f"- {t}\n"

    return r, hybrid


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE — always generates resume
# ═════════════════════════════════════════════════════════════════════════════

def process_resume(resume_file, job_url, job_desc_manual):
    if resume_file is None:
        yield "Upload your resume PDF to get started.", "", None
        return

    # Get job description
    job_desc = ""
    if job_desc_manual and len(job_desc_manual.strip()) > 50:
        job_desc = job_desc_manual.strip()
    elif job_url and job_url.strip():
        yield "Scraping job posting...", "", None
        job_desc = scrape_job_posting(job_url.strip())
        if job_desc.startswith("ERROR:"):
            yield f"Could not scrape URL. Paste the job description manually.\n\n`{job_desc}`", "", None
            return
    if not job_desc or len(job_desc) < 50:
        yield "Paste the job description (or a valid URL) to continue.", "", None
        return

    # Extract resume
    yield "Reading your resume...", "", None
    try:
        resume_text = extract_resume_text(resume_file.name)
    except Exception as e:
        yield f"Could not read PDF: {e}", "", None
        return
    if len(resume_text) < 50:
        yield "Could not extract text. Is this a scanned image?", "", None
        return

    # Embeddings
    yield "Computing semantic similarity...", "", None
    try:
        embed = compute_embedding_scores(resume_text, job_desc)
    except Exception as e:
        embed = {"overall_similarity": 0, "section_scores": [], "weak_coverage": []}

    # LLM keyword analysis
    yield "Analyzing ATS keywords with Llama 3.1-70B...", "", None
    try:
        analysis = analyze_keywords(resume_text, job_desc)
    except Exception as e:
        yield f"LLM analysis failed:\n```\n{traceback.format_exc()}\n```", "", None
        return

    report, hybrid = format_report(analysis, embed)

    # ALWAYS generate optimized resume (no skip gate)
    if hybrid < 40:
        yield report + "\n\n---\n*Low match — generating best possible optimization anyway...*", "", None
    else:
        yield report + "\n\n---\n*Generating optimized resume...*", "", None

    try:
        optimized_md = generate_optimized_resume(resume_text, job_desc, analysis)
    except Exception as e:
        yield report + f"\n\nResume generation failed: {e}", "", None
        return

    yield report + "\n\n---\n*Creating PDF...*", optimized_md, None
    try:
        pdf_path = tempfile.mktemp(suffix=".pdf")
        markdown_to_pdf(optimized_md, pdf_path)
    except Exception as e:
        yield report + f"\n\nPDF error: {e}", optimized_md, None
        return

    yield report + "\n\n---\n**Your optimized resume is ready below.**", optimized_md, pdf_path


# ═════════════════════════════════════════════════════════════════════════════
#  GRADIO UI — Polished single-column
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* Force light mode */
:root, .dark, .gradio-container, .gradio-container.dark {
    --body-background-fill: #f3f2ef !important;
    --body-text-color: #191919 !important;
    --block-background-fill: white !important;
    --block-border-color: #e8e7e4 !important;
    --block-label-text-color: #333 !important;
    --block-title-text-color: #191919 !important;
    --input-background-fill: #fafafa !important;
    --neutral-50: #fafafa !important;
    --neutral-100: #f5f5f5 !important;
    --neutral-200: #e5e5e5 !important;
    --neutral-300: #d4d4d4 !important;
    --neutral-400: #a3a3a3 !important;
    --neutral-500: #737373 !important;
    --neutral-600: #525252 !important;
    --neutral-700: #404040 !important;
    --neutral-800: #262626 !important;
    --neutral-900: #171717 !important;
    color-scheme: light !important;
}

/* Layout */
.gradio-container {
    max-width: 680px !important; margin: 0 auto !important;
    background: #f3f2ef !important; color: #191919 !important;
}
footer { display: none !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a66c2, #004182);
    border-radius: 12px; padding: 28px 28px 24px; margin-bottom: 16px; color: #fff;
}
.hero h1 { font-size: 1.4rem; font-weight: 700; margin: 0 0 6px; color: #fff !important; }
.hero p  { font-size: 0.82rem; color: rgba(255,255,255,.82); margin: 0; line-height: 1.55; }
.hero .tag {
    display: inline-block; margin-top: 10px; padding: 3px 10px;
    font-size: 0.7rem; font-weight: 500; color: rgba(255,255,255,.88);
    background: rgba(255,255,255,.13); border-radius: 16px;
}

/* Button — pill, NOT full-width */
button.primary {
    background: #0a66c2 !important; border: none !important;
    border-radius: 24px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 12px 48px !important;
    box-shadow: 0 2px 8px rgba(10,102,194,.25) !important;
    max-width: 260px !important; width: auto !important;
    display: block !important; margin: 0 auto 16px !important;
}
button.primary:hover {
    background: #004182 !important;
    box-shadow: 0 4px 14px rgba(10,102,194,.35) !important;
}

/* Labels */
label, span[data-testid="block-label"] {
    color: #333 !important; font-weight: 600 !important;
    font-size: 0.82rem !important; background: transparent !important;
}
.label-wrap span { background: transparent !important; }

/* Inputs */
textarea {
    border: 1.5px solid #ddd !important; border-radius: 8px !important;
    background: #fafafa !important; color: #191919 !important;
    font-size: 0.86rem !important;
}
textarea:focus {
    border-color: #0a66c2 !important;
    box-shadow: 0 0 0 3px rgba(10,102,194,.08) !important;
    background: #fff !important;
}

/* File upload — keep Gradio's default look, just refine */
.file-preview { border-radius: 8px !important; }

/* Accordion */
.label-wrap { cursor: pointer; }
.label-wrap span { color: #0a66c2 !important; font-weight: 500 !important; font-size: 0.84rem !important; }

/* Tables */
table { width: 100% !important; font-size: 0.84rem !important; border-collapse: collapse !important; }
table th {
    background: #f7f6f3 !important; color: #555 !important; font-weight: 600 !important;
    padding: 8px 12px !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: .04em;
    border-bottom: 2px solid #e8e7e4 !important; white-space: nowrap !important;
}
table td {
    padding: 8px 12px !important; color: #333 !important;
    border-bottom: 1px solid #f0efec !important;
}

/* Markdown report */
.prose, .markdown-text { color: #333 !important; font-size: 0.86rem !important; line-height: 1.6 !important; }
.prose h2 { color: #191919 !important; font-size: 1.02rem !important; font-weight: 700 !important; margin-top: 18px !important; padding-bottom: 6px; border-bottom: 2px solid #f0efec; }
.prose h3 { color: #333 !important; font-size: 0.9rem !important; font-weight: 600 !important; margin-top: 14px !important; }
.prose strong { color: #191919 !important; }
.prose hr { border-color: #eee !important; margin: 14px 0 !important; }

/* Footer */
.ft { text-align: center; font-size: 0.7rem; color: #aaa; padding: 16px 0 4px; }
.ft a { color: #0a66c2; text-decoration: none; font-weight: 500; }
"""

THEME = gr.themes.Default(
    primary_hue=gr.themes.Color(c50="#eef3f8", c100="#d0e0f0", c200="#a8c8e8",
        c300="#70a8d8", c400="#3d8ec8", c500="#0a66c2", c600="#0856a8",
        c700="#064a8f", c800="#043d75", c900="#03305c", c950="#022244"),
    neutral_hue="gray",
    font=("Inter", "system-ui", "sans-serif"),
).set(
    body_background_fill="#f3f2ef", body_background_fill_dark="#f3f2ef",
    body_text_color="#191919", body_text_color_dark="#191919",
    block_background_fill="white", block_background_fill_dark="white",
    block_border_color="#e8e7e4", block_border_color_dark="#e8e7e4",
    block_border_width="1px", block_border_width_dark="1px",
    block_shadow="0 1px 2px rgba(0,0,0,.06)", block_shadow_dark="0 1px 2px rgba(0,0,0,.06)",
    block_radius="10px",
    block_label_text_color="#333", block_label_text_color_dark="#333",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_title_text_color="#191919", block_title_text_color_dark="#191919",
    input_background_fill="#fafafa", input_background_fill_dark="#fafafa",
    input_border_color="#ddd", input_border_color_dark="#ddd",
    input_placeholder_color="#999", input_placeholder_color_dark="#999",
    button_primary_background_fill="#0a66c2", button_primary_background_fill_dark="#0a66c2",
    button_primary_background_fill_hover="#004182", button_primary_background_fill_hover_dark="#004182",
    button_primary_text_color="white", button_primary_text_color_dark="white",
    border_color_accent="#0a66c2", border_color_accent_dark="#0a66c2",
    color_accent_soft="#eef3f8", color_accent_soft_dark="#eef3f8",
)


with gr.Blocks(theme=THEME, css=CSS, title="ATS Resume Optimizer") as demo:

    # Hero
    gr.HTML("""<div class="hero">
        <h1>Resume Optimizer</h1>
        <p>Upload your resume and paste a job description. We score keyword match
        and semantic similarity, then rewrite your resume to maximize ATS compatibility.
        We never invent skills or experience.</p>
        <span class="tag">Powered by Llama 3.1-70B + Sentence-Transformers</span>
    </div>""")

    # Resume upload — full width, its own block
    resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"], type="filepath")

    # Job description — full width below
    job_desc_input = gr.Textbox(
        label="Job Description",
        placeholder="Paste the full job description here...",
        lines=6, max_lines=20,
    )

    # Hidden: job URL (user said it doesn't work, keep as hidden fallback)
    job_url_input = gr.Textbox(visible=False, value="")

    # Button — compact pill
    submit_btn = gr.Button("Analyze & Optimize", variant="primary")

    # Results
    report_output = gr.Markdown(value="", elem_id="report")

    # Optimized resume
    with gr.Accordion("Optimized Resume", open=False, visible=True):
        md_output = gr.Textbox(
            label="Markdown (editable)",
            lines=16, show_copy_button=True, interactive=True,
        )
        pdf_output = gr.File(label="Download PDF")

    submit_btn.click(
        fn=process_resume,
        inputs=[resume_input, job_url_input, job_desc_input],
        outputs=[report_output, md_output, pdf_output],
    )

    gr.HTML("""<div class="ft">
        Built by <a href="https://danielregaladoumiami.github.io/portfolio/">Daniel Regalado</a>
        &middot; University of Miami
        &middot; <a href="https://github.com/DanielRegaladoUMiami/ats-resume-optimizer">GitHub</a>
    </div>""")

if __name__ == "__main__":
    demo.launch()
