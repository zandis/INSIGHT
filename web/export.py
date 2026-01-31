"""
Export module for INSIGHT Web Application.

Provides export functionality for:
- JSON
- PDF
- CSV
- BibTeX
- Markdown
"""

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from web.database import ResearchSession


class ExportManager:
    """
    Manages export operations for research results.
    """

    MEDIA_TYPES = {
        "json": "application/json",
        "pdf": "application/pdf",
        "csv": "text/csv",
        "bibtex": "application/x-bibtex",
        "markdown": "text/markdown"
    }

    def get_media_type(self, format: str) -> str:
        """Get media type for export format."""
        return self.MEDIA_TYPES.get(format, "application/octet-stream")

    async def export(
        self,
        session: ResearchSession,
        results: List[Dict],
        format: str,
        include_citations: bool = True
    ) -> Any:
        """
        Export research results.

        Args:
            session: Research session
            results: Session results
            format: Export format
            include_citations: Include citations

        Returns:
            Export data in requested format
        """
        if format == "json":
            return await self._export_json(session, results, include_citations)
        elif format == "csv":
            return await self._export_csv(session, results)
        elif format == "bibtex":
            return await self.generate_bibtex(session.citations)
        elif format == "markdown":
            return await self._export_markdown(session, results, include_citations)
        elif format == "pdf":
            return await self._export_pdf(session, results, include_citations)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _export_json(
        self,
        session: ResearchSession,
        results: List[Dict],
        include_citations: bool
    ) -> Dict:
        """Export as JSON."""
        data = {
            "session_id": session.id,
            "objective": session.objective,
            "tools": session.tools,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "results": results,
            "key_findings": session.key_findings
        }

        if include_citations:
            data["citations"] = session.citations

        return data

    async def _export_csv(
        self,
        session: ResearchSession,
        results: List[Dict]
    ) -> bytes:
        """Export as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Task ID", "Content", "Created At"
        ])

        # Data rows
        for result in results:
            writer.writerow([
                result.get("task_id", ""),
                result.get("content", "")[:1000],  # Truncate long content
                result.get("created_at", "")
            ])

        return output.getvalue().encode('utf-8')

    async def _export_markdown(
        self,
        session: ResearchSession,
        results: List[Dict],
        include_citations: bool
    ) -> bytes:
        """Export as Markdown."""
        lines = [
            f"# INSIGHT Research Report",
            f"",
            f"**Objective:** {session.objective}",
            f"",
            f"**Tools Used:** {', '.join(session.tools)}",
            f"",
            f"**Date:** {session.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## Key Findings",
            f"",
        ]

        if session.key_findings:
            lines.append(session.key_findings)
        else:
            lines.append("No key findings available.")

        lines.extend([
            "",
            "## Detailed Results",
            ""
        ])

        for i, result in enumerate(results, 1):
            lines.extend([
                f"### Result {i}: {result.get('task_id', 'Unknown')}",
                "",
                result.get('content', 'No content'),
                ""
            ])

        if include_citations and session.citations:
            lines.extend([
                "---",
                "",
                "## Citations",
                ""
            ])
            for citation in session.citations:
                lines.append(f"- {self._format_citation(citation)}")

        return "\n".join(lines).encode('utf-8')

    async def _export_pdf(
        self,
        session: ResearchSession,
        results: List[Dict],
        include_citations: bool
    ) -> bytes:
        """
        Export as PDF.

        Note: Requires reportlab or similar library.
        Falls back to text if not available.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=20
            )
            story.append(Paragraph("INSIGHT Research Report", title_style))

            # Objective
            story.append(Paragraph(f"<b>Objective:</b> {session.objective}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Tools
            story.append(Paragraph(f"<b>Tools:</b> {', '.join(session.tools)}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Key Findings
            story.append(Paragraph("Key Findings", styles['Heading2']))
            if session.key_findings:
                # Clean HTML tags for PDF
                clean_findings = session.key_findings.replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(clean_findings[:5000], styles['Normal']))
            story.append(Spacer(1, 20))

            # Results summary
            story.append(Paragraph("Results Summary", styles['Heading2']))
            story.append(Paragraph(f"Total results: {len(results)}", styles['Normal']))

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            # Fallback to text
            return await self._export_markdown(session, results, include_citations)

    async def generate_bibtex(self, citations: List[Dict]) -> bytes:
        """
        Generate BibTeX from citations.

        Args:
            citations: List of citation dictionaries

        Returns:
            BibTeX formatted string
        """
        entries = []

        for i, citation in enumerate(citations):
            entry_type = citation.get("type", "article")
            key = citation.get("key", f"insight_{i+1}")

            fields = []
            for field in ["author", "title", "journal", "year", "volume",
                         "pages", "doi", "pmid", "abstract"]:
                if field in citation and citation[field]:
                    value = str(citation[field]).replace("{", "\\{").replace("}", "\\}")
                    fields.append(f"  {field} = {{{value}}}")

            entry = f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"
            entries.append(entry)

        return "\n\n".join(entries).encode('utf-8')

    def _format_citation(self, citation: Dict) -> str:
        """Format a single citation for display."""
        parts = []

        if "author" in citation:
            parts.append(citation["author"])

        if "title" in citation:
            parts.append(f'"{citation["title"]}"')

        if "journal" in citation:
            parts.append(f"*{citation['journal']}*")

        if "year" in citation:
            parts.append(f"({citation['year']})")

        if "doi" in citation:
            parts.append(f"DOI: {citation['doi']}")

        return " ".join(parts) if parts else str(citation)


class ReportGenerator:
    """
    Generate comprehensive research reports.
    """

    def __init__(self):
        """Initialize report generator."""
        self.export_manager = ExportManager()

    async def generate_summary_report(
        self,
        sessions: List[ResearchSession]
    ) -> Dict:
        """
        Generate summary report across multiple sessions.

        Args:
            sessions: List of sessions to summarize

        Returns:
            Summary report data
        """
        total_results = sum(s.results_count for s in sessions)
        completed = sum(1 for s in sessions if s.status == "completed")
        failed = sum(1 for s in sessions if s.status == "failed")

        # Aggregate tools usage
        tool_counts = {}
        for session in sessions:
            for tool in session.tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_sessions": len(sessions),
            "completed": completed,
            "failed": failed,
            "total_results": total_results,
            "tool_usage": tool_counts,
            "date_range": {
                "start": min(s.created_at for s in sessions).isoformat() if sessions else None,
                "end": max(s.created_at for s in sessions).isoformat() if sessions else None
            }
        }

    async def generate_comparison_report(
        self,
        sessions: List[ResearchSession],
        results_by_session: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Generate comparison report between sessions.

        Args:
            sessions: Sessions to compare
            results_by_session: Results for each session

        Returns:
            Comparison report data
        """
        comparison = {
            "sessions": [],
            "metrics": {
                "total_results": {},
                "tools_used": {},
                "duration": {}
            }
        }

        for session in sessions:
            session_data = {
                "id": session.id,
                "objective": session.objective,
                "status": session.status,
                "results_count": session.results_count,
                "tools": session.tools
            }
            comparison["sessions"].append(session_data)

            comparison["metrics"]["total_results"][session.id] = session.results_count
            comparison["metrics"]["tools_used"][session.id] = len(session.tools)

        return comparison
