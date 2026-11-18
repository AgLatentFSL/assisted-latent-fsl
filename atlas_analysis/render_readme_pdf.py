"""Render README.md to README.pdf (headless Chrome). Collapsible tables are listed by title only; their rows are in README.md and analysis/tables/.
Usage (from the atlas root): python analysis/render_readme_pdf.py"""
import html, os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); src = open(os.path.join(ROOT, "README.md")).read().split("\n")
def inl(s):
    s = html.escape(s, quote=False); s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s); s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f"<img src='file://{os.path.join(ROOT, m.group(2))}' alt='{m.group(1)}'>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<span class='lk'>\1</span>", s); s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s); s = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", s)
    return s.replace("&lt;sub&gt;", "<span class='tnote'>").replace("&lt;/sub&gt;", "</span>")
out, i, first_h1, folded = [], 0, True, []
while i < len(src):
    l = src[i]
    if l.startswith("<!--"): i += 1; continue
    if l.startswith("<details>"):
        t = ""
        while not src[i].startswith("</details>"):
            m = re.match(r"<summary>(.*)</summary>", src[i]); t = m.group(1) if m else t; i += 1
        folded.append(t); i += 1; continue
    if l.startswith("```"):
        j = i + 1; buf = []
        while not src[j].startswith("```"): buf.append(html.escape(src[j])); j += 1
        out.append("<pre>" + "\n".join(buf) + "</pre>"); i = j + 1; continue
    if l.startswith("#"):
        n = len(l) - len(l.lstrip("#")); t = inl(l[n:].strip())
        out.append((f"<div class='kicker'>Resource</div>" if n == 1 and first_h1 else "") + f"<h{n}>{t}</h{n}>"); first_h1 = False; i += 1; continue
    if l.startswith("|"):
        rows = []
        while i < len(src) and src[i].startswith("|"): rows.append([c.strip() for c in src[i].strip().strip("|").split("|")]); i += 1
        wide = len(rows[0]) > 9
        out.append(f"<table class='{'wide' if wide else ''}'><thead><tr>" + "".join(f"<th>{inl(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inl(c)}</td>" for c in r) + "</tr>" for r in rows[2:]) + "</tbody></table>"); continue
    if l.startswith("- "):
        items = []
        while i < len(src) and src[i].startswith("- "): items.append(f"<li>{inl(src[i][2:])}</li>"); i += 1
        out.append("<ul>" + "".join(items) + "</ul>"); continue
    if l.startswith("> "): out.append(f"<p class='abstract'>{inl(l[2:].replace('**Summary.** ', ''))}</p>"); i += 1; continue
    if l.strip():
        cls = " class='meta'" if (l.startswith("*") and l.endswith("*") and not l.startswith("**")) else (" class='cap'" if re.match(r"\*\*(Fig\.|Table) ", l) else "")
        out.append(f"<p{cls}>{inl(l)}</p>")
    i += 1
out.append(f"<p class='fold'>{len(folded)} tables are folded in <code>README.md</code> and are not printed here; every one is also a CSV file in <code>analysis/tables/</code>.</p>")
css = """@page{size:Letter;margin:16mm 15mm 15mm}*{box-sizing:border-box}body{font:9.4pt/1.5 Helvetica,Arial,sans-serif;color:#111;margin:0;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.kicker{font-size:8.5pt;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#0072B2;border-top:3pt solid #0072B2;padding-top:5pt;display:inline-block}
h1{font-size:19pt;line-height:1.17;margin:6pt 0 6pt}h2{font-size:12.5pt;margin:16pt 0 5pt;break-after:avoid;border-bottom:.6pt solid #999;padding-bottom:2pt}h3{font-size:10.5pt;margin:11pt 0 3pt;break-after:avoid}
.meta{font-size:8.6pt;color:#555;margin:0 0 9pt}.abstract{font-size:10.2pt;font-weight:700;margin:0 0 10pt}p{margin:0 0 7pt}.cap{font-size:8.5pt;margin:6pt 0 3pt;break-after:avoid}.fold{font-size:8.4pt;color:#444;border-left:2.5pt solid #bbb;padding-left:7pt}
ul{margin:0 0 8pt;padding-left:13pt}li{margin-bottom:3pt}code{font:8.2pt Menlo,monospace;background:#f1f2f4;padding:0 2pt;border-radius:1.5pt}.lk{color:#0072B2}
pre{font:7.8pt/1.5 Menlo,monospace;background:#f5f6f8;border-left:2.5pt solid #0072B2;padding:8pt 10pt;white-space:pre-wrap;break-inside:avoid;margin:4pt 0 9pt}
img{max-width:100%;max-height:205mm;display:block;margin:8pt auto 4pt}table{border-collapse:collapse;font-size:8.2pt;width:100%;border-top:1.2pt solid #111;border-bottom:1.2pt solid #111;margin:3pt 0 10pt;break-inside:avoid}
table.wide{font-size:6.6pt}th{text-align:left;padding:3.5pt 6pt 3.5pt 0;font-weight:700;border-bottom:.6pt solid #111;vertical-align:bottom}td{padding:2.3pt 6pt 2.3pt 0;vertical-align:top;font-variant-numeric:tabular-nums}.tnote{font-size:7.8pt;color:#444}"""
hp = os.path.join(ROOT, "analysis", "_readme.html"); open(hp, "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Layer-by-layer reference atlas</title><style>{css}</style></head><body>{''.join(out)}</body></html>")
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.path.join(ROOT, 'README.pdf')}", "file://" + hp], capture_output=True); os.remove(hp); print("README.pdf written")
