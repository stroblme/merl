# merl's chores. `make help` lists them.

PREFIX ?= $(HOME)/.local
CFLAGS ?= -O3

.PHONY: build install test clean help

all:
	$(MAKE) build
	$(MAKE) install

build: build/merl-rt

build/merl-rt: runtime/merl.c runtime/libsais.c runtime/libsais.h runtime/jsmn.h
	@mkdir -p build
	$(CC) $(CFLAGS) -pthread -o $@ runtime/merl.c runtime/libsais.c

install: build/merl-rt
	install -Dm755 build/merl-rt $(PREFIX)/bin/merl-rt

test:
	uv run pytest -q

clean:
	rm -rf build

help:
	@echo "build    compile the runtime to build/merl-rt (CFLAGS, default -O3)"
	@echo "install  copy it to PREFIX/bin/merl-rt (PREFIX, default ~/.local)"
	@echo "test     uv run pytest -q, the runtime pinned against merl.ngram"
	@echo "clean    remove build/"
