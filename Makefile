.PHONY: all clean

all: article.md
	pandoc article.md --template=template.tex -o output.pdf

article.md:
	cat \
		introduction.md \
		background.md \
		model_architecture.md \
		why_self_attention.md \
		training.md \
		results.md \
		conclusion.md \
		>> article.md

clean:
	rm -f article.md output.pdf