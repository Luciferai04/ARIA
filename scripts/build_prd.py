import re
import urllib.request
import base64
import json
import base64
from pathlib import Path
from md2pdf.core import md2pdf

# Path
md_path = Path("ARIA_Technical_PRD.md")
out_md = Path("ARIA_Technical_PRD_processed.md")
out_pdf = Path("ARIA_Technical_PRD.pdf")

content = md_path.read_text()

# CSS for styling the PDF beautifully (GitHub-like)
css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #24292e; background-color: #fff; padding: 20px; }
h1, h2, h3 { color: #0366d6; margin-top: 24px; margin-bottom: 16px; font-weight: 600; line-height: 1.25; }
h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
table { border-spacing: 0; border-collapse: collapse; margin-top: 0; margin-bottom: 16px; width: 100%; }
table th, table td { padding: 6px 13px; border: 1px solid #dfe2e5; }
table tr:nth-child(2n) { background-color: #f6f8fa; }
hr { height: 0.25em; padding: 0; margin: 24px 0; background-color: #e1e4e8; border: 0; }
pre { background-color: #f6f8fa; border-radius: 3px; padding: 16px; overflow: auto; }
code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 85%; padding: 0.2em 0.4em; margin: 0; background-color: rgba(27,31,35,0.05); border-radius: 3px; }
pre code { background-color: transparent; padding: 0; }
"""
Path("style.css").write_text(css)

# Find Mermaid blocks
def replacer(match):
    mermaid_code = match.group(1).strip()
    config = {
        "code": mermaid_code,
        "mermaid": {"theme": "default"}
    }
    encoded = base64.urlsafe_b64encode(json.dumps(config).encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/pako:{encoded}"
    
    # download to local to be safe with weasyprint
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open('architecture.jpg', 'wb') as f:
                f.write(response.read())
        return "![Architecture Diagram](architecture.jpg)"
    except Exception as e:
        print("Failed to download image", e)
        return match.group(0)

new_content = re.sub(r'```mermaid(.*?)```', replacer, content, flags=re.DOTALL)
out_md.write_text(new_content)

print(f"Generating PDF from processed Markdown...")
md2pdf(out_pdf,
       md_content=new_content,
       css_file_path="style.css",
       base_url=".")
print("Done!")
