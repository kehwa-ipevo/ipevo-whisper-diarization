
install-deps:
	pip install --no-cache-dir -r requirements.txt


.PHONY: build
build:
	sh build_exe.sh
