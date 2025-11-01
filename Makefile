.PHONY: all clean

all: introduction.md background.md model_architecture.md why_self_attention.md training.md results.md conclusion.md 
	pandoc $^ \
		--bibliography=references.bib \
		--citeproc \
		--pdf-engine=pdflatex \
		--template=template.tex \
		--csl=numeric \
		-o output.pdf

clean:
	rm -f output.pdf