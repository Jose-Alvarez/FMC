# Building the FMC manual

`FMC_manual.tex` is the **master document**. Edit it (not `README.md`, which is generated) and run

```bash
bash docs/build_manual.sh          # PDF and README.md
bash docs/build_manual.sh pdf      # only docs/FMC_manual.pdf
bash docs/build_manual.sh readme   # only README.md
```

| File | Purpose |
|------|---------|
| `FMC_manual.tex` | master document (the text of the manual) |
| `FMC_manual.bib` | bibliography (keys: first four letters of the first author + two digits of the year) |
| `figures/` | figures used by the manual (PNG, so that they can be shown in GitHub) |
| `build_manual.sh` | builds the PDF and the README |
| `tex2gfm.lua`, `unwrap_refs.lua` | pandoc filters that convert the LaTeX document to GitHub Markdown |
| `readme_header.md` | title, badges and links at the top of the README |
| `elsevier-harvard.csl` | citation style used in the README (the PDF uses `elsarticle-harv.bst`) |

The version and the date of the manual are set in the first lines of `FMC_manual.tex` (`\fmcversion`, `\fmcdate`) and are used in the PDF and in the README.

## Software required

* **bash** and **sed** (Linux and macOS have them; on Windows use WSL or Git Bash).
* **pandoc 3.0 or newer** (tested with 3.1.3). The citation processor and the Lua interpreter are included in pandoc.
* **A TeX distribution** (TeX Live, MacTeX or MiKTeX) with `pdflatex`, `bibtex` and `latexmk` (which needs Perl), and these packages: `natbib`, `hyperref`, `listings`, `longtable`, `booktabs`, `float`, `etoolbox`, `underscore`, `xcolor`, `amsmath`, `babel`, `mathpazo` and `avant` (PSNFSS fonts), `lmodern`, and the bibliography style `elsarticle-harv.bst` (package `elsarticle`).

Debian and Ubuntu:

```bash
sudo apt install pandoc latexmk lmodern texlive-latex-base texlive-latex-recommended \
                 texlive-fonts-recommended texlive-publishers
```

macOS: `brew install pandoc` and MacTeX (`brew install --cask mactex-no-gui`). Windows: pandoc, MiKTeX (it installs the missing packages when needed), Git for Windows (bash) and a Perl distribution such as Strawberry Perl. The simplest option on any system is a full TeX distribution (`texlive-full`, MacTeX).

Python, GMT, LyX and Ghostscript are **not** needed to build the manual (GMT is only used to regenerate the map figures).

## Rules for editing the master document

The PDF accepts any LaTeX. To obtain a good README the document must stay within the subset that pandoc understands:

* Structure with `\part`, `\section`, `\subsection`, `\subsubsection` and `\paragraph`. Put a `\label` after the headings that are referenced.
* Figures with the `figure` environment, `\includegraphics{figures/name.png}`, `\caption` and `\label`. Figures are numbered consecutively; use `\ref` to cite them. Sections and figures can be cited with `\ref`; other labels are not resolved in the README.
* Tables with `longtable` or `tabular` (columns `l` or `p{...\linewidth}`).
* Commands and results in `lstlisting` (add `[language=bash]` for commands).
* Lists with `itemize`, `enumerate` and `description`. In the README a description list becomes a bullet list with the term in bold.
* References with the `natbib` commands (`\citet`, `\citep`, `\citealp`) and the keys of `FMC_manual.bib`.
* Mathematics with `$...$`, `\[...\]`, `equation` and `align`. Displayed equations inside lists are written inline in the README.
* Do not redefine `\_` or other basic commands in the preamble: pandoc reads the preamble and would lose the characters.
* The license note of the title page is inside a `flushleft` environment, which is left out of the README. The unnumbered sections placed before the first `\part` (Acknowledgments) are moved to the end of the README.
* The Creative Commons badge of the title page is included if the file `figures/CreativeCommons_88x31.png` exists.
