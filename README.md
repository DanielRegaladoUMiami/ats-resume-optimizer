---
title: Resume Optimizer Agent
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: ATS-optimized resumes via Llama 3.1 + Sentence-Transformers
---

# 📄 Resume Optimizer Agent

An AI-powered tool that analyzes your resume against a specific job posting using a **hybrid ML approach**: LLM-based keyword extraction + embedding-based semantic similarity scoring.

## How It Works

1. **Upload** your resume (PDF) + paste a job posting URL or description
2. **Scraping** — auto-extracts the job description from the URL
3. **Semantic Similarity** — computes cosine similarity between resume and job embeddings (sentence-transformers / `all-MiniLM-L6-v2`)
4. **ATS Keyword Analysis** — Llama 3.1-70B extracts and matches ATS keywords
5. **Hybrid Score** — combines keyword match (60%) + semantic similarity (40%)
6. **Decision Gate:**
   - Score ≥ 50% → generates a tailored resume (Markdown + PDF)
   - Score < 50% → recommends skipping this role with gap analysis
7. **Output** — optimized resume in both editable Markdown and downloadable PDF

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Meta Llama 3.1-70B-Instruct (via HF Inference API) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Frontend | Gradio |
| PDF Processing | pdfplumber (read) + ReportLab (generate) |
| Web Scraping | BeautifulSoup4 |

## Honesty Guarantee

This tool will **never** invent skills, experience, or achievements. It only:
- Reorganizes sections to prioritize relevant experience
- Rephrases bullet points to include ATS keywords naturally
- Adjusts language to mirror the job description's terminology
- Reorders content to highlight the strongest matches

## Architecture

```
                    ┌─────────────┐
                    │  Resume PDF │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐     ┌──────────────┐
                    │  pdfplumber │     │  Job URL /    │
                    │  (extract)  │     │  Description  │
                    └──────┬──────┘     └──────┬───────┘
                           │                   │
                           │    ┌──────────────▼──────┐
                           │    │  BeautifulSoup      │
                           │    │  (scrape job post)  │
                           │    └──────────┬──────────┘
                           │               │
                    ┌──────▼───────────────▼──────┐
                    │        Analysis Engine       │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │ Sentence-Transformers  │  │
                    │  │ (semantic similarity)  │  │
                    │  └───────────┬────────────┘  │
                    │              │                │
                    │  ┌───────────▼────────────┐  │
                    │  │   Llama 3.1-70B        │  │
                    │  │   (keyword extraction  │  │
                    │  │    & resume rewrite)   │  │
                    │  └───────────┬────────────┘  │
                    │              │                │
                    │  ┌───────────▼────────────┐  │
                    │  │   Hybrid Scoring       │  │
                    │  │   60% KW + 40% embed   │  │
                    │  └───────────┬────────────┘  │
                    └──────────────┼────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Score >= 50%?            │
                    │   YES → Optimized Resume    │
                    │   NO  → Gap Analysis        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Output: Markdown + PDF    │
                    └─────────────────────────────┘
```

## Local Development

```bash
git clone https://github.com/DanielRegaladoUMiami/ats-resume-optimizer.git
cd ats-resume-optimizer
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
python app.py
```

## License

MIT

---

Built by [Daniel Regalado](https://danielregaladoumiami.github.io/portfolio/)
