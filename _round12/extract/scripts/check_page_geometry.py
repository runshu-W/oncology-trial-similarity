#!/usr/bin/env python3
"""Page-geometry check on the delivered PDFs (round-12).

The LaTeX log only reports overfull boxes for material TeX knows is too
wide; a tabular that overflows the text block can still be silently
clipped at the page edge in the delivered PDF.  This script uses
`pdftotext -bbox` to find, on every page, the right-most glyph extent
and flags any page whose content reaches beyond the text-block edge
(estimated as the median right-most extent over pages plus a small
tolerance) or beyond the physical page width.

Usage: check_page_geometry.py file.pdf [file.pdf ...]
Exit status 1 if any page overflows.
"""
import re
import subprocess
import sys
from statistics import median


def page_extents(pdf):
    xml = subprocess.run(["pdftotext", "-bbox", pdf, "-"],
                         capture_output=True, text=True, check=True).stdout
    pages = []
    for m in re.finditer(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>',
                         xml, re.S):
        w = float(m.group(1))
        xs = [float(x) for x in re.findall(r'xMax="([\d.]+)"', m.group(3))]
        pages.append((w, max(xs) if xs else 0.0))
    return pages


def main():
    bad = 0
    for pdf in sys.argv[1:]:
        pages = page_extents(pdf)
        med = median(x for _, x in pages)
        print(f"{pdf}: {len(pages)} pages; median right-most extent "
              f"{med:.1f}pt; page width {pages[0][0]:.0f}pt")
        for i, (w, x) in enumerate(pages, 1):
            if x > med + 3.0:
                flag = "BEYOND PAGE" if x > w else "beyond text block"
                print(f"  page {i}: right-most extent {x:.1f}pt ({flag})")
                bad += 1
    print("OK: no page overflows" if not bad else f"{bad} page(s) overflow")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
