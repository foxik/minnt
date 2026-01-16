.PHONY: all
all: site

prerequisities := Makefile mkdocs.yml $(wildcard src/* src/*/* template/partials/* ../minnt/* ../minnt/*/*)
site: get_katex $(prerequisities)
	venv/bin/mkdocs build
	@rm -rf site/assets/images site/assets/javascripts/lunr site/assets/javascripts/workers site/assets/*/*.map site/partials

.PHONY: install
install: site
	minnt_docs_upload site/

.PHONY: serve
serve: get_katex $(prerequisities)
	venv/bin/mkdocs serve -a localhost:1234

.PHONY: get_katex
get_katex: .cache/plugin/privacy/assets/external/cdn.jsdelivr.net/npm/katex/dist/katex.min.css
.cache/plugin/privacy/assets/external/cdn.jsdelivr.net/npm/katex/dist/katex.min.css:
	@mkdir -p $(dir $@)
	@curl https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css | sed 's#url(\([^)]*.woff2) format("woff2")\)[^}]*#url(https://cdn.jsdelivr.net/npm/katex/dist/\1#g' >$@

venv:
	python3 -m venv venv
	venv/bin/pip install --no-cache-dir black mkdocs-material mkdocstrings-python griffe-inherited-docstrings markdown-exec

.PHONY: clean
clean:
	rm -rf site/
