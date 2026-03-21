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


# ─── Config ───────────────────────────────────────────────────────────────────
LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
MATCH_THRESHOLD = 50

# ─── Initialize Models ───────────────────────────────────────────────────────
llm_client = InferenceClient(model=LLM_MODEL)
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
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        selectors = [
            "div.job-description", "div.jobsearch-jobDescriptionText",
            "div.description", "div.job-details", "article",
            "div.posting-requirements", "div[data-testid='job-description']",
            "div.job-posting", "section.job-description",
            "div#job-description", "div.jobDescriptionContent",
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return el.get_text(separator="\n", strip=True)

        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)[:8000]
        return ""
    except Exception as e:
        return f"ERROR: Could not scrape URL – {e}"


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
    """
    Compute semantic similarity between resume and job description
    using sentence-transformers embeddings.

    Returns:
        dict with overall_similarity, section_scores, and weak_coverage
    """
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
        return {
            "overall_similarity": round(overall_sim * 100, 1),
            "section_scores": [],
            "weak_coverage": [],
        }

    resume_embeddings = embed_model.encode(resume_chunks, convert_to_tensor=True)
    job_embeddings = embed_model.encode(job_chunks, convert_to_tensor=True)
    sim_matrix = util.cos_sim(resume_embeddings, job_embeddings)

    section_scores = []
    for i, chunk in enumerate(resume_chunks):
        max_sim = float(sim_matrix[i].max())
        preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
        section_scores.append({
            "section_preview": preview,
            "similarity": round(max_sim * 100, 1),
        })
    section_scores.sort(key=lambda x: x["similarity"], reverse=True)

    job_coverage = []
    for j, chunk in enumerate(job_chunks):
        max_sim = float(sim_matrix[:, j].max())
        preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
        job_coverage.append({
            "requirement_preview": preview,
            "best_match_score": round(max_sim * 100, 1),
        })
    job_coverage.sort(key=lambda x: x["best_match_score"])
    weak_areas = [j for j in job_coverage if j["best_match_score"] < 50]

    return {
        "overall_similarity": round(overall_sim * 100, 1),
        "section_scores": section_scores[:5],
        "weak_coverage": weak_areas[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM ANALYSIS (Llama 3.1-70B)
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = 3000) -> str:
    """Call Llama 3.1-70B via HF Inference API."""
    response = llm_client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
    )
    return response.strip()


def analyze_keywords(resume_text: str, job_description: str) -> dict:
    """Use LLM to extract and match ATS keywords."""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert ATS (Applicant Tracking System) analyst. You respond ONLY with valid JSON. No markdown, no backticks, no explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the resume against the job description.

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
{job_description[:3000]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""

    raw = "{" + call_llm(prompt, max_tokens=2500)

    raw = raw.strip()
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


def generate_optimized_resume(
    resume_text: str, job_description: str, analysis: dict
) -> str:
    """Use LLM to generate an optimized resume in markdown."""

    keywords_found = [
        k["keyword"]
        for k in analysis.get("ats_keywords", [])
        if k["status"] == "FOUND"
    ]
    suggestions = analysis.get("tailoring_suggestions", [])

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert resume writer specializing in ATS optimization. You write clean, professional resumes in Markdown format.

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
- Use bullet points for achievements
<|eot_id|><|start_header_id|>user<|end_header_id|>
Rewrite this resume to better match the target job.

TARGET: {analysis.get('job_title', 'N/A')} at {analysis.get('company', 'N/A')}
KEYWORDS TO INCORPORATE: {json.dumps(keywords_found)}
TAILORING SUGGESTIONS: {json.dumps(suggestions)}

ORIGINAL RESUME:
{resume_text[:4000]}

JOB DESCRIPTION (for context):
{job_description[:2000]}

Generate the optimized resume now in clean Markdown.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#"""

    result = "#" + call_llm(prompt, max_tokens=3000)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def markdown_to_pdf(markdown_text: str, output_path: str):
    """Convert optimized resume markdown to a clean PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ResumeName", parent=styles["Title"],
        fontSize=18, leading=22, textColor=HexColor("#1a1a2e"),
        spaceAfter=4, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "ResumeContact", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#555555"),
        alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=12, leading=15, textColor=HexColor("#1a1a2e"),
        spaceBefore=12, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"],
        fontSize=10, leading=13, leftIndent=18, bulletIndent=6,
        spaceBefore=1, spaceAfter=1, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#333333"),
        spaceBefore=6, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["Normal"],
        fontSize=10, leading=13, alignment=TA_JUSTIFY,
    ))

    story = []
    lines = markdown_text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        def clean_md(t):
            t = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"\*(.*?)\*", r"<i>\1</i>", t)
            return t

        if stripped.startswith("# ") and not stripped.startswith("## "):
            story.append(Paragraph(clean_md(stripped[2:].strip()), styles["ResumeName"]))
        elif stripped.startswith("## "):
            text = stripped[3:].strip().upper()
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=HexColor("#1a1a2e"), spaceAfter=2, spaceBefore=6
            ))
            story.append(Paragraph(clean_md(text), styles["SectionHead"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_md(stripped[4:].strip()), styles["SubHead"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(
                clean_md(stripped[2:].strip()), styles["BulletItem"], bulletText="•"
            ))
        elif "|" in stripped or "@" in stripped:
            story.append(Paragraph(clean_md(stripped), styles["ResumeContact"]))
        else:
            story.append(Paragraph(clean_md(stripped), styles["BodyText2"]))

    doc.build(story)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_analysis_report(analysis: dict, embed_scores: dict) -> tuple:
    """Format combined LLM + embedding analysis into a readable report."""
    llm_score = analysis.get("match_score", 0)
    embed_score = embed_scores.get("overall_similarity", 0)

    # Hybrid score: 60% keyword match + 40% semantic similarity
    hybrid_score = round(0.6 * llm_score + 0.4 * embed_score, 1)
    rec = analysis.get("recommendation", "N/A")

    if hybrid_score < MATCH_THRESHOLD:
        rec = "SKIP"
    elif hybrid_score >= MATCH_THRESHOLD and rec == "SKIP":
        rec = "PROCEED"

    if hybrid_score >= 75:
        indicator = "🟢"
    elif hybrid_score >= 50:
        indicator = "🟡"
    else:
        indicator = "🔴"

    report = f"""# 📊 ATS Match Analysis

## {analysis.get('job_title', 'Job')} @ {analysis.get('company', 'Company')}

---

### Hybrid Match Score: {indicator} {hybrid_score}%

| Component | Score | Weight |
|-----------|-------|--------|
| 🔑 ATS Keyword Match | {llm_score}% | 60% |
| 🧠 Semantic Similarity | {embed_score}% | 40% |
| **Combined** | **{hybrid_score}%** | |

**Recommendation: {"✅ PROCEED — Optimizing your resume" if rec == "PROCEED" else "⛔ SKIP — This role isn't a good match"}**

{analysis.get('recommendation_reason', '')}

---

### 🔑 ATS Keywords Scan

| Keyword | Status |
|---------|--------|
"""
    for kw in analysis.get("ats_keywords", []):
        status = "✅" if kw["status"] == "FOUND" else "❌"
        report += f"| {kw['keyword']} | {status} {kw['status']} |\n"

    if analysis.get("strengths"):
        report += "\n### 💪 Your Strengths\n"
        for s in analysis["strengths"]:
            report += f"- {s}\n"

    if analysis.get("critical_gaps"):
        report += "\n### ⚠️ Critical Gaps\n"
        for g in analysis["critical_gaps"]:
            report += f"- {g}\n"

    if embed_scores.get("section_scores"):
        report += "\n### 🧠 Semantic Analysis — Strongest Resume Sections\n"
        for sec in embed_scores["section_scores"][:3]:
            sim = sec["similarity"]
            bar = "█" * int(sim / 10) + "░" * (10 - int(sim / 10))
            report += f"- `{bar}` {sim}% — *{sec['section_preview'][:80]}...*\n"

    if embed_scores.get("weak_coverage"):
        report += "\n### 🔍 Job Requirements with Weak Resume Coverage\n"
        for wk in embed_scores["weak_coverage"]:
            report += f"- ⚠️ ({wk['best_match_score']}%) *{wk['requirement_preview'][:100]}...*\n"

    if rec == "PROCEED" and analysis.get("tailoring_suggestions"):
        report += "\n### 🎯 Tailoring Strategy\n"
        for t in analysis["tailoring_suggestions"]:
            report += f"- {t}\n"

    return report, hybrid_score, rec


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_resume(resume_file, job_url, job_description_manual):
    """Main orchestration function."""

    if resume_file is None:
        yield "❌ Please upload your resume (PDF).", None, None
        return

    job_description = ""

    if job_url and job_url.strip():
        yield "⏳ Scraping job posting...", None, None
        job_description = scrape_job_posting(job_url.strip())
        if job_description.startswith("ERROR:"):
            if job_description_manual and job_description_manual.strip():
                job_description = job_description_manual.strip()
            else:
                yield (
                    f"❌ {job_description}\n\n"
                    "**Tip:** Paste the job description manually in the text box below."
                ), None, None
                return

    if not job_description or len(job_description) < 100:
        if job_description_manual and job_description_manual.strip():
            job_description = job_description_manual.strip()
        else:
            yield "❌ Could not get job description. Please paste it manually.", None, None
            return

    yield "⏳ Reading your resume...", None, None
    resume_text = extract_resume_text(resume_file.name)
    if not resume_text or len(resume_text) < 50:
        yield "❌ Could not extract text from resume. Make sure it's not a scanned image.", None, None
        return

    yield "⏳ Computing semantic similarity (sentence-transformers)...", None, None
    try:
        embed_scores = compute_embedding_scores(resume_text, job_description)
    except Exception as e:
        embed_scores = {"overall_similarity": 0, "section_scores": [], "weak_coverage": []}
        print(f"Embedding error (non-fatal): {e}")

    yield "⏳ Analyzing ATS keywords (Llama 3.1-70B)...", None, None
    try:
        analysis = analyze_keywords(resume_text, job_description)
    except Exception as e:
        yield f"❌ LLM analysis failed: {e}", None, None
        return

    report, hybrid_score, rec = format_analysis_report(analysis, embed_scores)

    if rec == "SKIP":
        yield report, None, None
        return

    yield report + "\n\n⏳ Generating optimized resume...", None, None
    try:
        optimized_md = generate_optimized_resume(resume_text, job_description, analysis)
    except Exception as e:
        yield report + f"\n\n❌ Resume generation failed: {e}", None, None
        return

    yield report + "\n\n⏳ Creating PDF...", None, None
    try:
        pdf_path = tempfile.mktemp(suffix=".pdf")
        markdown_to_pdf(optimized_md, pdf_path)
    except Exception as e:
        yield report + f"\n\n⚠️ PDF generation issue: {e}", optimized_md, None
        return

    final_report = report + "\n\n---\n### ✅ Done! Download your optimized resume below."
    yield final_report, optimized_md, pdf_path


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Resume Optimizer Agent",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
    .main-header { text-align: center; margin-bottom: 10px; }
    .main-header h1 { font-size: 2em; margin-bottom: 5px; }
    .main-header p { color: #666; font-size: 1.1em; }
    .disclaimer { 
        background: #fff3cd; padding: 10px; border-radius: 8px; 
        border-left: 4px solid #ffc107; margin: 10px 0;
        font-size: 0.9em;
    }
    .tech-stack {
        background: #e8f4f8; padding: 8px 12px; border-radius: 8px;
        font-size: 0.85em; margin: 5px 0;
    }
    """
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>📄 Resume Optimizer Agent</h1>
        <p>ATS-optimized resumes tailored to specific job postings</p>
    </div>
    <div class="tech-stack">
        🧠 <strong>Powered by:</strong> Llama 3.1-70B (keyword analysis & rewriting) + 
        Sentence-Transformers (semantic similarity scoring) — fully open source
    </div>
    <div class="disclaimer">
        ⚡ <strong>Honesty guarantee:</strong> This tool will NEVER invent skills or experience. 
        It only reorganizes and rephrases what's already in your resume to match the job posting.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Your Inputs")
            resume_input = gr.File(
                label="Upload Resume (PDF)",
                file_types=[".pdf"],
                type="filepath",
            )
            job_url_input = gr.Textbox(
                label="Job Posting URL",
                placeholder="https://careers.example.com/job/12345",
                info="We'll scrape the job description automatically"
            )
            job_desc_input = gr.Textbox(
                label="Or Paste Job Description",
                placeholder="Paste the full job description here as fallback...",
                lines=8,
                info="Used if URL scraping fails or as primary input"
            )
            submit_btn = gr.Button(
                "🚀 Analyze & Optimize",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Report")
            report_output = gr.Markdown(
                value="*Upload your resume and provide a job posting to get started.*"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Optimized Resume (Markdown)")
            md_output = gr.Textbox(
                label="Editable Markdown",
                lines=20,
                show_copy_button=True,
                interactive=True,
            )
        with gr.Column():
            gr.Markdown("### 📄 Download PDF")
            pdf_output = gr.File(label="Optimized Resume PDF")

    submit_btn.click(
        fn=process_resume,
        inputs=[resume_input, job_url_input, job_desc_input],
        outputs=[report_output, md_output, pdf_output],
    )

    gr.Markdown("""
    ---
    <center>
    Built by <a href="https://danielregaladoumiami.github.io/portfolio/" target="_blank">Daniel Regalado</a> | 
    Powered by <a href="https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct" target="_blank">Llama 3.1-70B</a> + 
    <a href="https://www.sbert.net/" target="_blank">Sentence-Transformers</a>
    </center>
    """)

if __name__ == "__main__":
    demo.launch()
