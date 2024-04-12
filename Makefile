posts/%.html: posts/%.md
	pandoc $^ -s -o $@

index.md: posts/*.html
	python3 index.py > index.md

index.html: index.md
	pandoc index.md -s -o index.html
