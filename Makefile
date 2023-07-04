
.PHONY: build
build:
	docker build -t gtzan_serve .

.PHONY: serve
serve:
	docker run -p 8000:8000 --runtime=nvidia gtzan_serve