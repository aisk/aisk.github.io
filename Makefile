POSTS_MD := $(wildcard posts/*.md)
POSTS_HTML := $(POSTS_MD:.md=.html)

posts/%.html: posts/%.md
	pandoc $^ -s -o $@

index.md: $(POSTS_HTML) index.py
	python3 index.py > index.md

index.html: index.md
	pandoc index.md -s -o index.html
