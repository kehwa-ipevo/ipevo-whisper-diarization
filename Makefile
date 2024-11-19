
install-deps:
	pip install --no-cache-dir -r requirements.txt


.PHONY: build
build:
	sh build_exe.sh

benchmark:
	pip install flameprof
	python -m cProfile -o requests.prof diarize.py -a $(AUDIO)
	flameprof requests.prof > requests.svg

clean:
	rm -rdf ./temp_output*
	rm -rdf ./build
	rm -rdf ./dist
	rm -rdf ./requests.prof
	rm -rdf ./requests.svg