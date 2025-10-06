.PHONY: all
all: site

prerequisities := Makefile mkdocs.yml $(wildcard ../docs/* ../docs/*/*)
site: $(prerequisities)
	venv/bin/mkdocs build
	@rm -rf site/assets/images site/assets/javascripts/lunr site/assets/javascripts/workers site/assets/*/*.map

.PHONY: install
install: site
	minnt_docs_upload site/

.PHONY: serve
serve: $(prerequisities)
	venv/bin/mkdocs serve -a localhost:1234

.PHONY: clean
clean:
	rm -rf site/
