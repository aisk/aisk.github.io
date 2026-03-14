POSTS_MD := $(wildcard posts/*.md)
POSTS_HTML := $(POSTS_MD:.md=.html)
PANDOC_HEAD := head.html

posts/%.html: posts/%.md $(PANDOC_HEAD)
	pandoc $< -s -H $(PANDOC_HEAD) -o $@

index.md: $(POSTS_HTML) index.py
	python3 -u index.py > index.md

index.html: index.md $(PANDOC_HEAD)
	pandoc index.md -s -H $(PANDOC_HEAD) -o index.html
