"""PDF document loader for RAG knowledge ingestion."""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List

logger = logging.getLogger("rag.ingest.loaders.pdf")


def load_pdf_documents(filepath: Path) -> List[Dict[str, Any]]:
    """Loads knowledge documents from a PDF file."""
    if not filepath.is_file():
        return []

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed. Run `pip install pypdf` to process PDF files.")
        return []

    try:
        reader = PdfReader(str(filepath))
        text_pages = [p.extract_text() or "" for p in reader.pages]
        raw_full = "\n\n".join(text_pages)
        full_text = re.sub(r"[^\x00-\x7F]+", "-", raw_full).strip()
        if not full_text:
            logger.warning(f"No extractable text found in PDF: {filepath.name}")
            return []

        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        row_pattern = re.compile(
            r"^(\d+)\s+(.+?)\s+(\d{1,2}(?:st|nd|rd|th)?(?:\s*-\s*\d{1,2}(?:st|nd|rd|th)?)?\s+[A-Za-z]{3,}\s+\d{4})\s+(.+)$"
        )
        schedule_items = [row_pattern.match(l) for l in lines if row_pattern.match(l)]

        documents = []
        if schedule_items:
            headers = [l for l in lines if not row_pattern.match(l) and "s.no" not in l.lower()]
            overview_title = " - ".join(headers[:3]) if headers else filepath.stem

            documents.append({
                "id": f"pdf_{filepath.stem}_overview",
                "title": overview_title,
                "category": "schedule",
                "summary": f"Schedule and activity overview for {overview_title}.",
                "content": full_text,
                "aliases": [overview_title, filepath.stem],
                "keywords": ["schedule", "activities", "events", filepath.stem],
                "metadata": {"source": filepath.name},
                "is_active": True,
            })

            for m in schedule_items:
                s_no, act_type, planned_date, rest = m.groups()
                clean_title = re.sub(r"[^\x00-\x7F]+", "-", rest.strip()).strip("- ")
                title = f"{act_type}: {clean_title}" if clean_title else f"{act_type} #{s_no}"
                summary = f"{title} is scheduled for {planned_date}."
                documents.append({
                    "id": f"event_{filepath.stem}_{s_no}",
                    "title": title,
                    "category": "event",
                    "summary": summary,
                    "content": f"Event: {title}\nType: {act_type}\nDate: {planned_date}\nDetails: {m.group(0)}",
                    "aliases": [title, clean_title, act_type],
                    "keywords": ["event", act_type.lower(), filepath.stem],
                    "metadata": {"source": filepath.name, "s_no": s_no, "date": planned_date},
                    "is_active": True,
                })
        else:
            for idx, p_text in enumerate(text_pages):
                clean_p = re.sub(r"[^\x00-\x7F]+", "-", p_text).strip()
                if not clean_p:
                    continue
                p_lines = [l.strip() for l in clean_p.splitlines() if l.strip()]
                content_lines = [l for l in p_lines if not re.match(r"^.+?\|\s*Page\s*\d+$", l, re.IGNORECASE)]
                if content_lines:
                    first_line = content_lines[0]
                    if len(content_lines) > 1 and len(first_line) < 45 and not first_line.endswith("."):
                        title = f"{first_line} - {content_lines[1]}"
                    else:
                        title = first_line
                    summary = content_lines[1] if len(content_lines) > 1 else clean_p[:200]
                else:
                    title = p_lines[0] if p_lines else f"{filepath.stem} Part {idx + 1}"
                    summary = clean_p[:200]

                category = "report" if "report" in filepath.stem.lower() else "document"
                documents.append({
                    "id": f"pdf_{filepath.stem}_{idx + 1}",
                    "title": title[:120],
                    "category": category,
                    "summary": summary[:250],
                    "content": clean_p,
                    "aliases": [title[:120], filepath.stem],
                    "keywords": [category, "document", filepath.stem] + filepath.stem.replace("_", " ").split(),
                    "metadata": {"source": filepath.name, "page": idx + 1},
                    "is_active": True,
                })

        return documents
    except Exception as e:
        logger.error(f"Error parsing PDF {filepath.name}: {e}")
        return []
