# Resume Optimizer Agent — Deployment Guide for Cowork

## Overview

Deploy the **Resume Optimizer Agent** to two platforms:
1. **HuggingFace Space** (Gradio app)
2. **GitHub** (public repo)

All source files are included alongside this document.

---

## Files Included

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio application (Llama 3.1-70B + Sentence-Transformers) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation + HuggingFace Space metadata |
| `.gitignore` | Git ignore rules |
| `LICENSE` | MIT License |

---

## Task 1: Create GitHub Repository

### Repository Details
- **Owner:** `DanielRegaladoUMiami`
- **Repo name:** `resume-optimizer-agent`
- **Visibility:** Public
- **Description:** `ATS-optimized resumes tailored to job postings — Llama 3.1-70B + Sentence-Transformers hybrid ML scoring`
- **License:** MIT (already included as file)

### Steps
1. Go to https://github.com/new
2. Create the repo with the details above
3. Upload all 5 files (`app.py`, `requirements.txt`, `README.md`, `.gitignore`, `LICENSE`)
4. Commit message: `feat: initial release — resume optimizer agent with hybrid ML scoring`

---

## Task 2: Create HuggingFace Space

### Space Details
- **Owner:** `DanielRegaladoCardoso`
- **Space name:** `resume-optimizer-agent`
- **SDK:** Gradio
- **Hardware:** CPU Basic (free with Pro)
- **Visibility:** Public

### Steps
1. Go to https://huggingface.co/new-space
2. Fill in the details above
3. Upload all 5 files (`app.py`, `requirements.txt`, `README.md`, `.gitignore`, `LICENSE`)
4. Go to **Settings → Variables and Secrets**
5. Add a secret: `HF_TOKEN` with the HuggingFace API token (needed for Llama 3.1-70B Inference API access)

### Important Notes
- The `README.md` contains HuggingFace Space metadata in the YAML frontmatter (title, emoji, sdk, etc.) — this is required for the Space to configure correctly
- The app uses `meta-llama/Llama-3.1-70B-Instruct` via HF Inference API — the Pro subscription provides free access
- The `sentence-transformers` model (`all-MiniLM-L6-v2`) downloads automatically on first run

---

## Architecture Summary

```
User uploads PDF + Job URL
        │
        ▼
   Web Scraping (BeautifulSoup)
        │
        ▼
   ┌────────────────────────────┐
   │  Sentence-Transformers     │  ← ML Component
   │  (all-MiniLM-L6-v2)       │
   │  Cosine similarity score   │
   └────────────┬───────────────┘
                │
   ┌────────────▼───────────────┐
   │  Llama 3.1-70B             │  ← LLM Component
   │  ATS keyword extraction    │
   │  Resume rewriting          │
   └────────────┬───────────────┘
                │
   ┌────────────▼───────────────┐
   │  Hybrid Score              │
   │  60% keywords + 40% embed  │
   │  ≥50% → optimize resume   │
   │  <50% → skip, show gaps   │
   └────────────┬───────────────┘
                │
                ▼
   Output: Markdown + PDF
```

---

## Tech Stack
- **LLM:** Meta Llama 3.1-70B-Instruct (HF Inference API, free with Pro)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Frontend:** Gradio
- **PDF Read:** pdfplumber
- **PDF Write:** ReportLab
- **Scraping:** BeautifulSoup4
