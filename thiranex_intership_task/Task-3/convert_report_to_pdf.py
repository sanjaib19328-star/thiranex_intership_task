"""
Script to convert EDA_Project_Report.md into a publication-grade, beautifully styled PDF.
Embeds high-resolution charts, styled tables, badges, callouts, and clean page breaks.
"""

import os
import re
import base64
import subprocess
import time

def encode_image_base64(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as img_file:
            return "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")
    return ""

# Read markdown content
with open("EDA_Project_Report.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# Pre-encode images to base64
image_map = {
    "images/01_hotel_type_distribution.png": encode_image_base64("images/01_hotel_type_distribution.png"),
    "images/02_cancellation_rates_by_hotel.png": encode_image_base64("images/02_cancellation_rates_by_hotel.png"),
    "images/03_monthly_booking_trends.png": encode_image_base64("images/03_monthly_booking_trends.png"),
    "images/04_top_guest_countries.png": encode_image_base64("images/04_top_guest_countries.png"),
    "images/05_market_segments_and_channels.png": encode_image_base64("images/05_market_segments_and_channels.png"),
    "images/06_length_of_stay_distribution.png": encode_image_base64("images/06_length_of_stay_distribution.png"),
    "images/07_lead_time_vs_cancellation.png": encode_image_base64("images/07_lead_time_vs_cancellation.png"),
    "images/08_adr_monthly_seasonality.png": encode_image_base64("images/08_adr_monthly_seasonality.png"),
    "images/09_deposit_and_customer_types.png": encode_image_base64("images/09_deposit_and_customer_types.png"),
    "images/10_repeated_guests_and_history.png": encode_image_base64("images/10_repeated_guests_and_history.png"),
    "images/11_parking_and_special_requests.png": encode_image_base64("images/11_parking_and_special_requests.png"),
    "images/12_correlation_heatmap.png": encode_image_base64("images/12_correlation_heatmap.png"),
}

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Comprehensive EDA Project Report: Hotel Booking Demand Analysis</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    @page {
        size: A4;
        margin: 20mm 18mm 22mm 18mm;
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
            font-family: 'Inter', sans-serif;
            font-size: 8.5pt;
            color: #64748B;
        }
        @bottom-left {
            content: "Thiranex Data Science Internship • Task 3 EDA Report";
            font-family: 'Inter', sans-serif;
            font-size: 8.5pt;
            color: #64748B;
        }
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
        background-color: #FFFFFF;
        line-height: 1.62;
        font-size: 10pt;
        margin: 0;
        padding: 0;
    }

    /* Cover / Header Section */
    .report-header {
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 20px;
        margin-bottom: 28px;
    }

    .badge-container {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 8pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-primary { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-success { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-warning { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }

    h1 {
        font-size: 22pt;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.25;
        margin: 8px 0 14px 0;
        letter-spacing: -0.5px;
    }

    .meta-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 9pt;
    }

    .meta-item strong {
        color: #0F172A;
        font-weight: 600;
    }

    /* Headings */
    h2 {
        font-size: 14pt;
        font-weight: 700;
        color: #1E3A8A;
        border-bottom: 1.5px solid #E2E8F0;
        padding-bottom: 6px;
        margin-top: 28px;
        margin-bottom: 12px;
        letter-spacing: -0.3px;
        page-break-after: avoid;
    }

    h3 {
        font-size: 11.5pt;
        font-weight: 700;
        color: #0F172A;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }

    h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #334155;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }

    p {
        margin: 0 0 10px 0;
        text-align: justify;
    }

    ul, ol {
        margin: 0 0 12px 0;
        padding-left: 22px;
    }

    li {
        margin-bottom: 4px;
    }

    /* Executive Callouts & Alerts */
    .callout {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 14px 0;
        page-break-inside: avoid;
    }

    .callout-title {
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 4px;
        font-size: 9.5pt;
    }

    .callout-warning {
        background: #FEF2F2;
        border-left-color: #EF4444;
    }
    .callout-warning .callout-title { color: #B91C1C; }

    .callout-success {
        background: #F0FDF4;
        border-left-color: #10B981;
    }
    .callout-success .callout-title { color: #047857; }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }

    th {
        background: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        text-align: left;
        padding: 7px 10px;
        border: 1px solid #334155;
    }

    td {
        padding: 6px 10px;
        border: 1px solid #E2E8F0;
        color: #334155;
    }

    tr:nth-child(even) td {
        background-color: #F8FAFC;
    }

    /* Images and Figures */
    .figure-box {
        text-align: center;
        margin: 18px 0;
        page-break-inside: avoid;
    }

    .figure-box img {
        max-width: 96%;
        height: auto;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .figure-caption {
        font-size: 8pt;
        color: #64748B;
        margin-top: 6px;
        font-style: italic;
    }

    /* Code & Monospace */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: #F1F5F9;
        color: #0F172A;
        padding: 1.5px 4.5px;
        border-radius: 4px;
        font-size: 8pt;
        border: 1px solid #E2E8F0;
    }

    /* Page Breaks */
    .page-break {
        page-break-before: always;
    }

    .avoid-break {
        page-break-inside: avoid;
    }
</style>
</head>
<body>

<div class="report-header">
    <div class="badge-container">
        <span class="badge badge-primary">Thiranex Data Science Internship</span>
        <span class="badge badge-success">Task 3 Formal Deliverable</span>
        <span class="badge badge-warning">Hospitality Analytics</span>
    </div>
    <h1>Comprehensive EDA Project Report:<br>Hotel Booking Demand Analysis</h1>
    <div style="font-size: 11pt; color: #475569; font-weight: 500;">
        Analytical Reasoning, Empirical Hypothesis Testing, and Revenue Management Intelligence
    </div>
    <div class="meta-grid">
        <div class="meta-item"><strong>Author:</strong> Sanjai B (Data Science Intern)</div>
        <div class="meta-item"><strong>Organization:</strong> Thiranex Internship Program</div>
        <div class="meta-item"><strong>Dataset:</strong> Kaggle Hotel Booking Demand (119,390 records, 32 attributes)</div>
        <div class="meta-item"><strong>Technologies:</strong> Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, SciPy</div>
    </div>
</div>

<div class="callout callout-success">
    <div class="callout-title">Executive Summary</div>
    <p style="margin: 0; font-size: 9pt;">
        Hotel booking cancellations, volatile demand seasonality, and unpredictable guest behaviors pose persistent challenges to hospitality revenue management and operational planning. This report delivers an in-depth exploratory data analysis (EDA) of <strong>119,390 real-world hotel booking records</strong> from two contrasting properties in Portugal: an urban <strong>City Hotel</strong> (Lisbon) and a coastal <strong>Resort Hotel</strong> (Algarve), collected across a 26-month observation period from July 2015 to August 2017. Moving beyond surface-level descriptive cleaning, this study applies analytical reasoning to uncover the core economic, operational, and behavioral drivers governing booking cancellations, pricing elasticity (ADR), geographic feeder markets, and customer micro-commitments.
    </p>
</div>
"""

# Process the markdown body into HTML
# Replace markdown table syntax with HTML tables
# Replace markdown headers with HTML headers
# Replace images with base64 embedded figure boxes

def markdown_to_html(md_text):
    # Split content starting from Section 1
    sec1_match = re.search(r'## 1\. Introduction', md_text)
    if sec1_match:
        body = md_text[sec1_match.start():]
    else:
        body = md_text

    # Replace section headers to add page breaks where logical
    sections_to_break = [
        "## 4. Data Understanding",
        "## 6. Univariate Analysis",
        "## 7. Bivariate Analysis",
        "## 8. Correlation Analysis",
        "## 10. Major Patterns and Trends",
        "## 11. Key Insights",
        "## 12. Conclusion"
    ]
    for sec in sections_to_break:
        if sec in body:
            body = body.replace(sec, f'<div class="page-break"></div>\n{sec}')

    # Replace images with embedded figure boxes
    for img_path, b64 in image_map.items():
        pattern = rf'!\[(.*?)\]\({re.escape(img_path)}\)'
        replacement = f'''<div class="figure-box">
    <img src="{b64}" alt="\\1">
    <div class="figure-caption">\\1</div>
</div>'''
        body = re.sub(pattern, replacement, body)

    # Convert headers
    body = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', body, flags=re.M)
    body = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', body, flags=re.M)
    body = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', body, flags=re.M)

    # Convert bold and italics
    body = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body)
    body = re.sub(r'\*(.*?)\*', r'<em>\1</em>', body)

    # Convert inline code
    body = re.sub(r'`(.*?)`', r'<code>\1</code>', body)

    # Convert Markdown tables to HTML tables
    lines = body.split('\n')
    in_table = False
    table_lines = []
    output_lines = []

    def format_table(tbl_lines):
        if len(tbl_lines) < 2:
            return '\n'.join(tbl_lines)
        headers = [c.strip() for c in tbl_lines[0].strip('|').split('|')]
        html_tbl = ['<table>', '<thead>', '<tr>']
        for h in headers:
            html_tbl.append(f'<th>{h}</th>')
        html_tbl.extend(['</tr>', '</thead>', '<tbody>'])

        for row in tbl_lines[2:]: # skip separator
            if not row.strip():
                continue
            cells = [c.strip() for c in row.strip('|').split('|')]
            html_tbl.append('<tr>')
            for c in cells:
                html_tbl.append(f'<td>{c}</td>')
            html_tbl.append('</tr>')

        html_tbl.extend(['</tbody>', '</table>'])
        return '\n'.join(html_tbl)

    for line in lines:
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                output_lines.append(format_table(table_lines))
                table_lines = []
                in_table = False
            output_lines.append(line)
    if in_table:
        output_lines.append(format_table(table_lines))

    processed = '\n'.join(output_lines)

    # Convert unordered lists
    processed = re.sub(r'^\* (.*?)$', r'<li>\1</li>', processed, flags=re.M)
    processed = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', processed)

    # Convert paragraphs
    final_lines = []
    for block in processed.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('<h') or block.startswith('<table') or block.startswith('<div') or block.startswith('<ul') or block.startswith('<ol'):
            final_lines.append(block)
        else:
            final_lines.append(f'<p>{block}</p>')

    return '\n\n'.join(final_lines)

html_body = markdown_to_html(md_content)

full_html = html_template + html_body + "\n</body>\n</html>"

with open("EDA_Project_Report.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("Saved self-contained EDA_Project_Report.html with base64 embedded charts.")

# Generate PDF via Chrome Headless
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html_abs = os.path.abspath("EDA_Project_Report.html")
pdf_abs = os.path.abspath("EDA_Project_Report.pdf")

if os.path.exists(pdf_abs):
    try:
        os.remove(pdf_abs)
    except Exception:
        pass

print("Generating EDA_Project_Report.pdf via Chrome Headless...")
cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_abs}",
    html_abs
]

res = subprocess.run(cmd, capture_output=True, text=True)
time.sleep(2) # Give file handle moment to release

if os.path.exists(pdf_abs):
    size_mb = os.path.getsize(pdf_abs) / (1024 * 1024)
    print(f"SUCCESS: EDA_Project_Report.pdf generated successfully! (File Size: {size_mb:.2f} MB)")
else:
    print("PDF generation failed. Output:", res.stderr)
