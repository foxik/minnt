.PHONY: all
all: build

.PHONY: build
build:
	make -C .. site
	rsync -a --del ../site/ site
	sed '/ doc-heading"/{h;s/.*id="\([^"]*\)".*/\1/;x}; / doc-\(class\|function\|attribute\)-name"/{G;s/\(<span [^>]*>\)\(.*\)\n\(.*\)/\1<span style="display:none">\3<\/span>\2/}' -i site/*.html site/*/*.html site/*/*/*.html
	cd site && ../dashing build -f ../dashing.json
	rm -rf minnt.docset
	mv -f site/minnt.docset .
	cp -a icon.png icon@2x.png minnt.docset/

.PHONY: clean
clean:
	rm -rf minnt.docset site
