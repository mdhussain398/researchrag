"""
Custom CSS design system and typography tokens for ResearchRAG.
"""

CUSTOM_CSS = """
<style>
/* Base Theme & Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Main Container Adjustments */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1200px;
}

/* Metric Cards */
.metric-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 12px;
}
.metric-card-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
}
.metric-card-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 4px;
}
.metric-card-subtitle {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Grounding & Verification Badges */
.badge-verified {
    display: inline-block;
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-partial {
    display: inline-block;
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-unverified {
    display: inline-block;
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Evidence Cards */
.evidence-card {
    background-color: #ffffff;
    border-left: 4px solid #2563eb;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.evidence-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #475569;
    font-weight: 600;
    margin-bottom: 6px;
}
.evidence-text {
    font-size: 0.9rem;
    color: #1e293b;
    line-height: 1.5;
}

/* Contradiction Box */
.contradiction-box {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #dc2626;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* Research Gap Box */
.gap-box {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* Custom Table Polish */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.styled-table th {
    background-color: #f1f5f9;
    color: #334155;
    font-weight: 600;
    text-align: left;
    padding: 10px;
    border-bottom: 2px solid #cbd5e1;
}
.styled-table td {
    padding: 9px 10px;
    border-bottom: 1px solid #e2e8f0;
    color: #1e293b;
}

/* Sidebar Customization */
.sidebar .sidebar-content {
    background-color: #f8fafc;
}
</style>
"""
