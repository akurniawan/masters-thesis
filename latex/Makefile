clean:
	rm -f sections/*.log sections/*.dvi sections/*.aux sections/*.toc sections/*.lof sections/*.lot sections/*.out sections/*.bbl sections/*.blg sections/*.xmpi
	rm -f cuni/*.log cuni/*.dvi cuni/*.aux cuni/*.toc cuni/*.lof cuni/*.lot cuni/*.out cuni/*.bbl cuni/*.blg cuni/*.xmpi cuni/*.fls cuni/*.fdb_latexmk cuni/*.synctex.gz
	rm -f malta/*.log malta/*.dvi malta/*.aux malta/*.toc malta/*.lof malta/*.lot malta/*.out malta/*.bbl malta/*.blg malta/*.xmpi malta/*.fls malta/*.fdb_latexmk malta/*.synctex.gz
	rm pdf/*.pdf

final: build-all
	mv cuni/thesis.pdf pdf/cuni-thesis.pdf
	mv cuni/abstract.pdf pdf/cuni-abstract.pdf
	mv malta/fyp.pdf pdf/malta-thesis.pdf

build-all:
	$(MAKE) -C cuni all
	$(MAKE) -C malta all

link:
	ln -sF /Volumes/Workspace/masters/thesis/latex/sections /Volumes/Workspace/masters/thesis/latex/cuni/sections
	ln -sF /Volumes/Workspace/masters/thesis/latex/sections /Volumes/Workspace/masters/thesis/latex/malta/sections
	ln -sF /Volumes/Workspace/masters/thesis/latex/img /Volumes/Workspace/masters/thesis/latex/cuni/img
	ln -sF /Volumes/Workspace/masters/thesis/latex/img /Volumes/Workspace/masters/thesis/latex/malta/img
	ln -s /Volumes/Workspace/masters/thesis/latex/bibliography.bib /Volumes/Workspace/masters/thesis/latex/cuni/bibliography.bib
	ln -s /Volumes/Workspace/masters/thesis/latex/bibliography.bib /Volumes/Workspace/masters/thesis/latex/malta/bibliography.bib

unlink:
	unlink /Volumes/Workspace/masters/thesis/latex/cuni/sections
	unlink /Volumes/Workspace/masters/thesis/latex/malta/sections
	unlink /Volumes/Workspace/masters/thesis/latex/cuni/img
	unlink /Volumes/Workspace/masters/thesis/latex/malta/img
	unlink /Volumes/Workspace/masters/thesis/latex/cuni/bibliography.bib
	unlink /Volumes/Workspace/masters/thesis/latex/malta/bibliography.bib

save:
	git add .
	git commit
	git push origin main	