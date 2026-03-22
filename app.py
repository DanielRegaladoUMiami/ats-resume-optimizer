"""
Resume Optimizer Agent
======================
ATS-optimized resumes tailored to job postings.
Uses Llama 3.1-70B (HF Inference) + Sentence-Transformers for hybrid ML scoring.

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
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


# ─── Config ──────────────────────────────────────────────────────────────────
LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
MATCH_THRESHOLD = 50

# ─── Initialize Models ──────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN", "")
llm_client = InferenceClient(model=LLM_MODEL, token=hf_token)
embed_model = SentenceTransformer(EMBED_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
#  WEB SCRAPING
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_job_posting(url: str) -> str:
    """Scrape job description from a URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        selectors = [
            "div.job-description", "div.jobsearch-jobDescriptionText",
            "div.description", "div.job-details", "article",
            "div.posting-requirements", "div[data-testid='job-description']",
            "div.job-posting", "section.job-description",
            "div#job-description", "div.jobDescriptionContent",
            "div.show-more-less-html__markup", "div.description__text",
            "main", "div[role='main']",
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return el.get_text(separator="\n", strip=True)[:8000]

        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)[:8000]
            if len(text) > 200:
                return text
        return ""
    except Exception as e:
        return f"ERROR: Could not scrape URL - {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_resume_text(pdf_path: str) -> str:
    """Extract text from uploaded resume PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING-BASED SIMILARITY SCORE (ML COMPONENT)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_embedding_scores(resume_text: str, job_description: str) -> dict:
    resume_embedding = embed_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = embed_model.encode(job_description, convert_to_tensor=True)
    overall_sim = float(util.cos_sim(resume_embedding, job_embedding)[0][0])

    def chunk_text(text, chunk_size=200):
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
            if len(words[i:i + chunk_size]) > 20
        ]

    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_description, chunk_size=100)

    if not resume_chunks or not job_chunks:
        return {"overall_similarity": round(overall_sim * 100, 1), "section_scores": [], "weak_coverage": []}

    resume_embeddings = embed_model.encode(resume_chunks, convert_to_tensor=True)
    job_embeddings = embed_model.encode(job_chunks, convert_to_tensor=True)
    sim_matrix = util.cos_sim(resume_embeddings, job_embeddings)

    section_scores = []
    for i, chunk in enumerate(resume_chunks):
        max_sim = float(sim_matrix[i].max())
        preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
        section_scores.append({"section_preview": preview, "similarity": round(max_sim * 100, 1)})
    section_scores.sort(key=lambda x: x["similarity"], reverse=True)

    job_coverage = []
    for j, chunk in enumerate(job_chunks):
        max_sim = float(sim_matrix[:, j].max())
        preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
        job_coverage.append({"requirement_preview": preview, "best_match_score": round(max_sim * 100, 1)})
    job_coverage.sort(key=lambda x: x["best_match_score"])
    weak_areas = [j for j in job_coverage if j["best_match_score"] < 50]

    return {
        "overall_similarity": round(overall_sim * 100, 1),
        "section_scores": section_scores[:5],
        "weak_coverage": weak_areas[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM ANALYSIS (Llama 3.1-70B via chat_completion)
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(system_msg: str, user_msg: str, max_tokens: int = 3000) -> str:
    """Call Llama 3.1-70B via HF Inference API using chat_completion."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    response = llm_client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def analyze_keywords(resume_text: str, job_description: str) -> dict:
    """Use LLM to extract and match ATS keywords."""
    system_msg = (
        "You are an expert ATS (Applicant Tracking System) analyst. "
        "You respond ONLY with valid JSON. No markdown, no backticks, no explanation."
    )
    user_msg = f"""Analyze the resume against the job description.

INSTRUCTIONS:
1. Extract the top 15-20 ATS keywords from the job description (hard skills, tools, certifications, methodologies)
2. For each keyword, check if it appears (or a close synonym) in the resume
3. Calculate match percentage: (found keywords / total keywords) * 100
4. List critical missing keywords
5. List the candidate's strengths relevant to this role
6. If score >= 50: recommend PROCEED with tailoring suggestions
7. If score < 50: recommend SKIP with explanation

Respond with ONLY this JSON (no other text):
{{"job_title": "...", "company": "...", "ats_keywords": [{{"keyword": "...", "status": "FOUND or MISSING", "resume_evidence": "brief quote or null"}}], "match_score": 75, "critical_gaps": ["..."], "recommendation": "PROCEED or SKIP", "recommendation_reason": "...", "tailoring_suggestions": ["..."], "strengths": ["..."]}}

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:3000]}"""

    raw = call_llm(system_msg, user_msg, max_tokens=2500)

    # Extract JSON from response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # Find the outermost JSON object
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in LLM response: {raw[:200]}")
    raw = raw[start:]

    brace_count = 0
    end_idx = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    if end_idx > 0:
        raw = raw[:end_idx]

    return json.loads(raw)


def generate_optimized_resume(resume_text: str, job_description: str, analysis: dict) -> str:
    """Use LLM to generate an optimized resume in markdown."""
    keywords_found = [k["keyword"] for k in analysis.get("ats_keywords", []) if k["status"] == "FOUND"]
    suggestions = analysis.get("tailoring_suggestions", [])

    system_msg = """You are an expert resume writer specializing in ATS optimization. You write clean, professional resumes in Markdown format.

CRITICAL RULES:
- NEVER invent or fabricate any experience, skill, project, or achievement
- ONLY use information that EXISTS in the original resume
- You may reorganize, reword, and emphasize existing content
- You may rephrase bullet points to naturally include relevant keywords
- You may reorder sections to prioritize the most relevant experience
- You may adjust language to mirror the job description terminology
- Keep the resume to 1-2 pages of content
- Start with the candidate name as # heading
- Use ## for section headers, ### for job titles/schools
- Use bullet points for achievements"""

    user_msg = f"""Rewrite this resume to better match the target job.

TARGET: {analysis.get('job_title', 'N/A')} at {analysis.get('company', 'N/A')}
KEYWORDS TO INCORPORATE: {json.dumps(keywords_found)}
TAILORING SUGGESTIONS: {json.dumps(suggestions)}

ORIGINAL RESUME:
{resume_text[:4000]}

JOB DESCRIPTION (for context):
{job_description[:2000]}

Generate the optimized resume now in clean Markdown."""

    return call_llm(system_msg, user_msg, max_tokens=3000)


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def markdown_to_pdf(markdown_text: str, output_path: str):
    """Convert optimized resume markdown to a clean PDF."""
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ResumeName", parent=styles["Title"], fontSize=18, leading=22, textColor=HexColor("#1a1a2e"), spaceAfter=4, alignment=TA_CENTER))
    styles.add(ParagraphStyle("ResumeContact", parent=styles["Normal"], fontSize=9, leading=12, textColor=HexColor("#555555"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle("SectionHead", parent=styles["Heading2"], fontSize=12, leading=15, textColor=HexColor("#1a1a2e"), spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle("BulletItem", parent=styles["Normal"], fontSize=10, leading=13, leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=1, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle("SubHead", parent=styles["Normal"], fontSize=10, leading=13, textColor=HexColor("#333333"), spaceBefore=6, spaceAfter=2))
    styles.add(ParagraphStyle("BodyText2", parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_JUSTIFY))

    def clean_md(t):
        t = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.*?)\*", r"<i>\1</i>", t)
        return t

    story = []
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
        elif stripped.startswith("# ") and not stripped.startswith("## "):
            story.append(Paragraph(clean_md(stripped[2:].strip()), styles["ResumeName"]))
        elif stripped.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#1a1a2e"), spaceAfter=2, spaceBefore=6))
            story.append(Paragraph(clean_md(stripped[3:].strip().upper()), styles["SectionHead"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_md(stripped[4:].strip()), styles["SubHead"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(clean_md(stripped[2:].strip()), styles["BulletItem"], bulletText="•"))
        elif "|" in stripped or "@" in stripped:
            story.append(Paragraph(clean_md(stripped), styles["ResumeContact"]))
        else:
            story.append(Paragraph(clean_md(stripped), styles["BodyText2"]))
    doc.build(story)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_analysis_report(analysis: dict, embed_scores: dict) -> tuple:
    llm_score = analysis.get("match_score", 0)
    embed_score = embed_scores.get("overall_similarity", 0)
    hybrid_score = round(0.6 * llm_score + 0.4 * embed_score, 1)
    rec = analysis.get("recommendation", "N/A")

    if hybrid_score < MATCH_THRESHOLD:
        rec = "SKIP"
    elif hybrid_score >= MATCH_THRESHOLD and rec == "SKIP":
        rec = "PROCEED"

    indicator = "🟢" if hybrid_score >= 75 else ("🟡" if hybrid_score >= 50 else "🔴")

    report = f"""## {analysis.get('job_title', 'Job')} at {analysis.get('company', 'Company')}

### {indicator} Match Score: {hybrid_score}%

| Component | Score | Weight |
|-----------|-------|--------|
| ATS Keyword Match | {llm_score}% | 60% |
| Semantic Similarity | {embed_score}% | 40% |
| **Combined** | **{hybrid_score}%** | |

**{"Optimizing your resume..." if rec == "PROCEED" else "This role may not be the best match."}**

{analysis.get('recommendation_reason', '')}

---

### ATS Keywords

| Keyword | Status |
|---------|--------|
"""
    for kw in analysis.get("ats_keywords", []):
        icon = "+" if kw["status"] == "FOUND" else "-"
        report += f"| {kw['keyword']} | {icon} {kw['status']} |\n"

    if analysis.get("strengths"):
        report += "\n### Your Strengths\n"
        for s in analysis["strengths"]:
            report += f"- {s}\n"

    if analysis.get("critical_gaps"):
        report += "\n### Gaps to Address\n"
        for g in analysis["critical_gaps"]:
            report += f"- {g}\n"

    if embed_scores.get("section_scores"):
        report += "\n### Semantic Analysis\n"
        for sec in embed_scores["section_scores"][:3]:
            sim = sec["similarity"]
            report += f"- {sim}% match — *{sec['section_preview'][:80]}...*\n"

    if embed_scores.get("weak_coverage"):
        report += "\n### Weak Coverage Areas\n"
        for wk in embed_scores["weak_coverage"]:
            report += f"- {wk['best_match_score']}% — *{wk['requirement_preview'][:100]}...*\n"

    if rec == "PROCEED" and analysis.get("tailoring_suggestions"):
        report += "\n### Tailoring Strategy\n"
        for t in analysis["tailoring_suggestions"]:
            report += f"- {t}\n"

    return report, hybrid_score, rec


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_resume(resume_file, job_url, job_description_manual):
    """Main orchestration function."""
    if resume_file is None:
        yield "Please upload your resume (PDF).", None, None
        return

    job_description = ""

    # Try manual input first (more reliable), then URL
    if job_description_manual and job_description_manual.strip() and len(job_description_manual.strip()) > 50:
        job_description = job_description_manual.strip()
    elif job_url and job_url.strip():
        yield "Scraping job posting...", None, None
        job_description = scrape_job_posting(job_url.strip())
        if job_description.startswith("ERROR:"):
            yield f"{job_description}\n\nPaste the job description manually instead.", None, None
            return

    if not job_description or len(job_description) < 50:
        yield "Please paste the job description or provide a valid URL.", None, None
        return

    yield "Reading your resume...", None, None
    try:
        resume_text = extract_resume_text(resume_file.name)
    except Exception as e:
        yield f"Could not read PDF: {e}", None, None
        return

    if not resume_text or len(resume_text) < 50:
        yield "Could not extract text from resume. Make sure it's not a scanned image.", None, None
        return

    yield "Computing semantic similarity...", None, None
    try:
        embed_scores = compute_embedding_scores(resume_text, job_description)
    except Exception as e:
        embed_scores = {"overall_similarity": 0, "section_scores": [], "weak_coverage": []}
        print(f"Embedding error (non-fatal): {e}")

    yield "Analyzing ATS keywords with Llama 3.1-70B...", None, None
    try:
        analysis = analyze_keywords(resume_text, job_description)
    except Exception as e:
        yield f"LLM analysis failed: {e}\n\n```\n{traceback.format_exc()}\n```", None, None
        return

    report, hybrid_score, rec = format_analysis_report(analysis, embed_scores)

    if rec == "SKIP":
        yield report, None, None
        return

    yield report + "\n\nGenerating optimized resume...", None, None
    try:
        optimized_md = generate_optimized_resume(resume_text, job_description, analysis)
    except Exception as e:
        yield report + f"\n\nResume generation failed: {e}", None, None
        return

    yield report + "\n\nCreating PDF...", None, None
    try:
        pdf_path = tempfile.mktemp(suffix=".pdf")
        markdown_to_pdf(optimized_md, pdf_path)
    except Exception as e:
        yield report + f"\n\nPDF issue: {e}", optimized_md, None
        return

    yield report + "\n\n---\n**Done! Download your optimized resume below.**", optimized_md, pdf_path


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI — LinkedIn-inspired design
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
* { font-family: 'Inter', -apple-system, system-ui, sans-serif !important; }
.gradio-container { max-width: 1060px !important; margin: 0 auto !important; background: #f4f2ee !important; }
footer { display: none !important; }

/* Header bar */
.top-bar {
    background: white;
    border-bottom: 1px solid #e0dfdc;
    padding: 12px 24px;
    margin: -16px -16px 16px -16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.top-bar .logo {
    width: 34px; height: 34px;
    background: #0a66c2;
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 18px;
}
.top-bar h1 { font-size: 1.1rem !important; font-weight: 600 !important; color: #191919 !important; margin: 0 !important; }
.top-bar p { font-size: 0.78rem; color: #666; margin: 0; }

/* Cards */
.card {
    background: white;
    border: 1px solid #e0dfdc;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.card-header { font-size: 0.95rem; font-weight: 600; color: #191919; margin-bottom: 8px; }
.card-sub { font-size: 0.78rem; color: #666; margin-bottom: 12px; }

/* Info banner */
.info-banner {
    background: #eef3f8;
    border: 1px solid #d0e0f0;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #333;
    margin-bottom: 8px;
}
.info-banner strong { color: #0a66c2; }

/* Override Gradio blocks to look like LinkedIn cards */
.block { background: white !important; border: 1px solid #e0dfdc !important; border-radius: 8px !important; }
.panel { background: white !important; }

/* Buttons */
button.primary {
    background: #0a66c2 !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
}
button.primary:hover { background: #004182 !important; }

/* Tabs */
.tab-nav { background: white !important; border-bottom: 1px solid #e0dfdc !important; border-radius: 8px 8px 0 0 !important; }
.tab-nav button { color: #666 !important; font-weight: 500 !important; font-size: 0.85rem !important; }
.tab-nav button.selected { color: #0a66c2 !important; border-bottom-color: #0a66c2 !important; font-weight: 600 !important; }

/* Input styling */
input, textarea, .wrap { border-color: #e0dfdc !important; border-radius: 6px !important; }
input:focus, textarea:focus { border-color: #0a66c2 !important; box-shadow: 0 0 0 1px #0a66c2 !important; }
label, .label-wrap span { color: #191919 !important; font-weight: 500 !important; font-size: 0.85rem !important; }

/* Tables */
table { border-collapse: collapse !important; }
table thead th { background: #f8f8f6 !important; color: #191919 !important; font-weight: 600 !important; font-size: 0.82rem !important; border-bottom: 2px solid #e0dfdc !important; }
table tbody td { color: #333 !important; font-size: 0.82rem !important; background: white !important; border-bottom: 1px solid #f0efec !important; }
table tbody tr:hover td { background: #f8f8f6 !important; }

/* Markdown */
.prose, .markdown-text { color: #333 !important; font-size: 0.88rem !important; line-height: 1.5 !important; }
.prose strong, .markdown-text strong { color: #191919 !important; }
.prose h2, .markdown-text h2 { color: #191919 !important; font-size: 1.15rem !important; }
.prose h3, .markdown-text h3 { color: #191919 !important; font-size: 1rem !important; }

/* File upload */
.upload-button { border-radius: 6px !important; border: 1px dashed #ccc !important; }

/* Footer */
.ft { text-align: center; font-size: 0.72rem; color: #999; padding: 12px 0 4px; }
.ft a { color: #0a66c2; text-decoration: none; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eef3f8", c100="#d0e0f0", c200="#a8c8e8",
        c300="#70a8d8", c400="#3d8ec8", c500="#0a66c2",
        c600="#0856a8", c700="#064a8f", c800="#043d75",
        c900="#03305c", c950="#022244",
    ),
    secondary_hue="slate",
    neutral_hue="slate",
    font=("Inter", "-apple-system", "system-ui", "sans-serif"),
).set(
    body_background_fill="#f4f2ee",
    body_background_fill_dark="#f4f2ee",
    body_text_color="#191919",
    body_text_color_dark="#191919",
    block_background_fill="white",
    block_background_fill_dark="white",
    block_border_color="#e0dfdc",
    block_border_color_dark="#e0dfdc",
    block_label_text_color="#666",
    block_label_text_color_dark="#666",
    block_title_text_color="#191919",
    block_title_text_color_dark="#191919",
    input_background_fill="white",
    input_background_fill_dark="white",
    input_border_color="#e0dfdc",
    input_border_color_dark="#e0dfdc",
    button_primary_background_fill="#0a66c2",
    button_primary_background_fill_dark="#0a66c2",
    button_primary_background_fill_hover="#004182",
    button_primary_background_fill_hover_dark="#004182",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    border_color_primary="#e0dfdc",
    border_color_primary_dark="#e0dfdc",
)


with gr.Blocks(theme=THEME, css=CSS, title="ATS Resume Optimizer") as demo:

    # Header
    gr.HTML("""
    <div class="top-bar">
        <div class="logo">R</div>
        <div>
            <h1>Resume Optimizer</h1>
            <p>ATS-optimized resumes powered by Llama 3.1-70B + Sentence-Transformers</p>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="info-banner">
        <strong>How it works:</strong> Upload your resume PDF + paste a job description.
        We analyze ATS keyword matches (60%) and semantic similarity (40%) to score your fit,
        then rewrite your resume to maximize your match. <strong>We never invent skills or experience.</strong>
    </div>
    """)

    with gr.Row(equal_height=False):
        # Left column — inputs
        with gr.Column(scale=1, min_width=320):
            resume_input = gr.File(
                label="Resume (PDF)",
                file_types=[".pdf"],
                type="filepath",
            )
            job_desc_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the full job description here...",
                lines=10,
                info="Copy-paste from the job posting for best results"
            )
            job_url_input = gr.Textbox(
                label="Or Job URL (optional)",
                placeholder="https://...",
                info="We'll try to scrape it — paste text above if this fails"
            )
            submit_btn = gr.Button("Analyze & Optimize", variant="primary", size="lg")

        # Right column — results
        with gr.Column(scale=2):
            report_output = gr.Markdown(
                value="Upload your resume and paste a job description to get started."
            )

    with gr.Row():
        with gr.Column():
            md_output = gr.Textbox(
                label="Optimized Resume (Markdown)",
                lines=18,
                show_copy_button=True,
                interactive=True,
            )
        with gr.Column():
            pdf_output = gr.File(label="Download Optimized PDF")

    submit_btn.click(
        fn=process_resume,
        inputs=[resume_input, job_url_input, job_desc_input],
        outputs=[report_output, md_output, pdf_output],
    )

    gr.HTML("""
    <div class="ft">
        Built by <a href="https://danielregaladoumiami.github.io/portfolio/">Daniel Regalado</a> · University of Miami ·
        <a href="https://github.com/DanielRegaladoUMiami/ats-resume-optimizer">GitHub</a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
