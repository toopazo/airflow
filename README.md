# airflow

Airflow is a pet project showing my work on **Face Recgnition from video**. This repo hold the code, installation and instructions. The idea behind this project is explained [in this article](https://toopazo.github.io/face-recognition-challenge/) (in Spanish) of [my website](https://toopazo.github.io).

## Installation

### Database (Postgres)
I followed this excellent tutorial on [how to use the official Postgres' Docker image](https://www.docker.com/blog/how-to-use-the-postgres-docker-official-image/)

The ```.env``` file should have the following variables
```bash
POSTGRES_USER=your-db-user
POSTGRES_PASSWORD=your-db-pass
POSTGRES_DB=airflow
COMPOSE_EXPOSED_PORT=54322
```

## Getting started

To run the example, first ..

## Others
### Side project ZDG

Imagen base toopazo/zdg

Esta imagen es creada a partir de la excelente libreria ZMQ para intercomunicar los contenedores siguiendo un grafo dirigido.

Build the image using
```bash
  docker build -t toopazo/zdg -f Dockerfile .
```

Build the image using
```bash
docker build -t toopazo/zdg -f Dockerfile .
```
Run the image using
 ```bash 
docker run -it zdg
docker run -it zdg bash
```

Remove image using
```bash
  docker rmi -f 42af9b40137d
```

Remove all stopped containers
```bash
docker rm $(docker ps --filter status=exited -q)
```

Remove dangling images (those called <none>)
```bash
docker rmi -f $(docker images -f "dangling=true" -q)
```

Stop and remove all containers
```bash
docker ps -aq | xargs docker stop | xargs docker rm
```

Export this image to file
```bash
docker save zdg > docker_image.tar 
```

Load this image to docker
```bash
docker load --input docker_image.tar
```
