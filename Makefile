ORG ?= zakazglobal
IMAGE_API=ghcr.io/$(ORG)/qg-bilingual-api
IMAGE_DEMO=ghcr.io/$(ORG)/qg-bilingual-demo
TAG ?= latest
CPU_TAG=$(TAG)-cpu
CUDA_TAG=$(TAG)-cuda

build-cpu:
	docker build -f docker/Dockerfile.cpu -t $(IMAGE_API):$(CPU_TAG) .
	docker build -f docker/Dockerfile.cpu -t $(IMAGE_DEMO):$(CPU_TAG) .

build-cuda:
	docker build -f docker/Dockerfile.cuda -t $(IMAGE_API):$(CUDA_TAG) .
	docker build -f docker/Dockerfile.cuda -t $(IMAGE_DEMO):$(CUDA_TAG) .

run-api:
	docker compose up -d api

run-demo:
	docker compose up -d demo

push:
	docker push $(IMAGE_API):$(CPU_TAG)
	docker push $(IMAGE_DEMO):$(CPU_TAG)

push-cuda:
	docker push $(IMAGE_API):$(CUDA_TAG)
	docker push $(IMAGE_DEMO):$(CUDA_TAG)

down:
	docker compose down
