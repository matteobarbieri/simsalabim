.PHONY: mongo-up
mongo-up:
	docker compose up

.PHONY: mongo-down
mongo-down:
	docker compose down

.PHONY: build
build:
	docker build --build-arg CHKPT_PATH=$(CHKPT_PATH) -t gtzan_serve .

.PHONY: serve
serve:
	docker run -p 8000:8000 --runtime=nvidia gtzan_serve