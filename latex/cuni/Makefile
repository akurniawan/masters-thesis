# export TEXINPUTS=../tex//:

all: thesis.pdf abstract.pdf abstractcs.pdf
# LaTeX must be run multiple times to get references right
thesis.pdf: thesis.tex $(wildcard *.tex) bibliography.bib thesis.xmpdata
	pdflatex $<
	bibtex thesis
	pdflatex $<
	pdflatex $<

abstract.pdf: abstract.tex abstract.xmpdata
	pdflatex $<

abstractcs.pdf: abstract_cs.tex abstract_cs.xmpdata
	pdflatex $<