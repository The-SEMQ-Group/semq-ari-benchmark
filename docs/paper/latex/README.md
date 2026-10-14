# Submission build

`ari.tex` is the only paper source in this checkout. The removed Markdown and HTML
drafts are preserved in Git history, but are not maintained copies.

## Build

    pdflatex ari.tex && pdflatex ari.tex

No TEXINPUTS or other tricks needed. Two passes: the page-count check uses a
`\label` that resolves on the second run.

    grep -o "newlabel{bodyend}{{[0-9]*}{[0-9]*}" ari.aux

`bodyend` is the page the body ends on. The CFP limit is 4 pages excluding
references and appendices, so this must read 4 or less.

## Figures

Three scripts in this directory write the PDFs `ari.tex` includes:

    python make_figure1.py   # figure1.pdf: SciFact gap (inline) + trajectory compounding (hash-checked archive)
    python make_figure2.py   # figure2.pdf: hosted decoding panel, reads experiments/arid-dry-run/results/
    python make_figure3.py   # figure3.pdf: batch invariance, reads experiments/batch-invariance/results/

They need `matplotlib` and `numpy` only. `make_figure1.py` also needs the
`L3_09_agentic_compounding` bundle under `../evidence/`; it is not committed, and
the script stops with a message pointing to `../evidence/README.md` for the fetch
instructions. `make_figure2.py` selects the published
row for each cell by transcript name, because `analysis.json` also holds prefix
smokes and confounder arms that are not the published numbers.

## Toolchain

Earlier builds failed because this TeX Live (2024) was missing `environ.sty`,
`trimspaces.sty` and `lineno.sty`, and `tlmgr` refused to install them: the
mirrors now serve the 2026 release and cross-release updates are blocked. The
real packages were fetched from CTAN and installed into

    ~/Library/TinyTeX/texmf-local/tex/latex/{environ,trimspaces,lineno}/

so `pdflatex` finds them. Nothing is stubbed any more.

## Style file

`neurips_2026.sty` is the official NeurIPS **2025** file with three strings
changed locally: package name, `\@neuripsordinal` 39th to 40th, and
`\@neuripsyear` 2025 to 2026. NeurIPS has still not published a 2026 Styles.zip
(`media.neurips.cc/Conferences/NeurIPS2026/Styles.zip` is a 404). Swap in the
official file when it appears. Recheck layout and page count against the official style; future geometry is not guaranteed.

## Line numbers

The `final` option is what produces the correct workshop footer, and it also
suppresses line numbers. Dropping `final` from the `\usepackage` line gives you
line numbers but changes the footer to "Submitted to 40th Conference ... Do not
distribute." The CFP does not require line numbers. Pick one:

    \usepackage[final,sglblindworkshop]{neurips_2026}   % workshop footer, no line numbers (current)
    \usepackage[sglblindworkshop]{neurips_2026}         % line numbers, "Submitted to" footer

## Revision notes

The revised source leads with the measurement protocol; the abstract is unchanged.
It is an independent draft, not a replacement for the submitted workshop PDF;
check `bodyend` after building rather than assuming the four-page limit still holds.
Two-pass compilation verified `bodyend` on page 4. Retrieval stress-test details
and archive limitations appear in the appendix.
