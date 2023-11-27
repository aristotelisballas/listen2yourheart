SHELL := /bin/bash

include env.sh


all: clean build


build:
	docker build -t ${IMAGE_NAME} .


clean:
	docker image rm ${IMAGE_NAME}

stop:
	docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}
