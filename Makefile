SHELL := /bin/bash

run:
	go run main.go

bench:
	go test -bench . ./... --benchmem
