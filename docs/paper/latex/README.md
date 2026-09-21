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
official file when it appears. Geometry does not change year to year, so the
page count will hold.

## Line numbers

The `final` option is what produces the correct workshop footer, and it also
suppresses line numbers. Dropping `final` from the `\usepackage` line gives you
line numbers but changes the footer to "Submitted to 40th Conference ... Do not
distribute." The CFP does not require line numbers. Pick one:

    \usepackage[final,sglblindworkshop]{neurips_2026}   % workshop footer, no line numbers (current)
    \usepackage[sglblindworkshop]{neurips_2026}         % line numbers, "Submitted to" footer

## Venue requirements

Checked 2026-09-16 against <http://mlforsystems.org/call_for_papers.html> (ML for
Systems workshop at NeurIPS 2026). The release checklist in
[docs/RELEASE.md](../../RELEASE.md) points here.

| Requirement | Value on the CFP page |
| --- | --- |
| Page limit | "submissions of up to 4 pages, not including references or Appendices. This year, this is a strict limit." |
| Format | "should follow the NeurIPS 2026 format" |
| File type | "All submissions must be in PDF format" |
| Anonymization | "Submissions do not have to be anonymized." |
| Submission site | OpenReview, `NeurIPS.cc/2026/Workshop/MLForSys` |
| Deadline | "August 29, 2026 by midnight (Anywhere on Earth)"; a second line on the page reads "Saturday August 29ths Monday August 31th, 2026". |
| Workshop | NeurIPS 2026, December 11 or 12 (TBA), International Convention Center Sydney |
| Proceedings | "Accepted papers will be optionally linked on the workshop website, but there will be no formal proceedings." Authors may publish the work elsewhere. |
