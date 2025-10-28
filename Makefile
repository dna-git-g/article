.PHONY: all clean

all: article.md
	pandoc article.md --template=template.tex -o output.pdf

article.md:
	cat \
		introduction.md \
		>> article.md

clean:
	rm -f article.md output.pdf