# DevOps

## Installation

Please install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

Please install VSCode extension(s):

- Docker

To build local docker image, open Docker Desktop and from `./devops` folder run the command:

```bash
docker build -t openisr:0.0.0 -f Dockerfile ..
```

To clean docker cache:

```bash
docker builder prune -a
```
