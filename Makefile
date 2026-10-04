POSTS_MD := $(wildcard posts/*.md)
POSTS_HTML := $(POSTS_MD:.md=.html)
PANDOC_HEAD := head.html
PANDOC_BACK := back.html

posts/%.html: posts/%.md $(PANDOC_HEAD) $(PANDOC_BACK)
	pandoc $< -s -M lang=en -H $(PANDOC_HEAD) -B $(PANDOC_BACK) -o $@

index.md: $(POSTS_HTML) index.goblin
	goblin run index.goblin > index.md

index.html: index.md $(PANDOC_HEAD)
	pandoc index.md -s -M lang=en -H $(PANDOC_HEAD) -o index.html
