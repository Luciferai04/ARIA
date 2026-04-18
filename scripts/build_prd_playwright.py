import markdown
import asyncio
from playwright.async_api import async_playwright
import re
import json
import base64

def process_mermaid(html):
    def replacer(match):
        code = match.group(1).strip()
        # Decode HTML entities
        code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        config = {
            "code": code,
            "mermaid": {"theme": "default"}
        }
        encoded = base64.urlsafe_b64encode(json.dumps(config).encode('utf-8')).decode('utf-8')
        return f'<img src="https://mermaid.ink/img/{encoded}" style="max-width: 100%; display: block; margin: 20px auto; border: 1px solid #eee; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">'
        
    return re.sub(r'<code class="language-mermaid">(.*?)</code>', replacer, html, flags=re.DOTALL)

async def main():
    with open('ARIA_Technical_PRD.md', 'r') as f:
        md_text = f.read()

    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    html_content = process_mermaid(html_content)

    css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        color: #1f2937;
        line-height: 1.6;
        padding: 40px;
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
      }
      h1, h2, h3 { color: #4f46e5; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-top: 32px; font-weight: 700; }
      h1 { font-size: 32px; border-bottom: 4px solid #4f46e5; padding-bottom: 12px; }
      h2 { font-size: 24px; }
      table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; page-break-inside: avoid; }
      th, td { border: 1px solid #e5e7eb; padding: 12px 16px; text-align: left; }
      th { background-color: #f9fafb; font-weight: 600; color: #4b5563; }
      tr:nth-child(even) { background-color: #f9fafb; }
      pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
      pre { background-color: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); }
      code { font-size: 13px; background-color: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
      pre code { background-color: transparent; padding: 0; border-radius: 0; }
      img { max-width: 100%; height: auto; }
      hr { border: 0; height: 1px; background: #e5e7eb; margin: 32px 0; }
      ul, ol { padding-left: 24px; }
      @media print {
        @page { margin: 20mm; }
        body { padding: 0; max-width: none; }
        h1, h2, h3, h4, h5 { page-break-after: avoid; }
        table, pre, img { page-break-inside: avoid; }
      }
    </style>
    """

    final_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{css}</head><body>{html_content}</body></html>"

    with open('temp.html', 'w') as f:
        f.write(final_html)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        import os
        await page.goto(f"file://{os.path.abspath('temp.html')}", wait_until="networkidle")
        await page.pdf(
            path='ARIA_Technical_PRD.pdf',
            format='A4',
            print_background=True,
            display_header_footer=True,
            header_template='<div></div>',
            footer_template='<div style="font-size:10px; width:100%; text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
            margin={"top": "in", "right": "in", "bottom": "in", "left": "in"}
        )
        await browser.close()
    
    print("PDF successfully generated using Playwright.")

asyncio.run(main())
