PDF_LATEX_ARGS=-halt-on-error -output-directory=release/target -shell-escape

.PHONY: all mkdir build clean cleanall

# by default, build the main documents
all: build

# create a directory to store all files
mkdir:
	mkdir -p release/target

# build tex files at the root directory
build: mkdir
	latexmk ${PDF_LATEX_ARGS} -pdf main.tex
	cp release/target/main.pdf .

# standard clean build files
clean:
	latexmk -c

# clean all generated files
cleanall: clean
	latexmk -C
	rm -rf release
	rm -f $(wildcard *.aux *.bbl *fdb_latexmk *.fls *.log *.ptc *.synctex.gz)
	rm -f $(wildcard contents/*.aux)
