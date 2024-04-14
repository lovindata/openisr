# DevOps

## Installation

Please install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

Please install VSCode extension(s):

- Docker

## Local build

To build local docker image:

```bash
docker build -t lovindata/openisr:local -f Dockerfile ..
```

To clean docker cache:

```bash
docker builder prune -af
```

## Remote multi-platform build

To list, create, use and delete the multi-platform builder:

```bash
docker buildx ls
```

```bash
docker buildx create --use --name multi-platform-builder
```

```bash
docker buildx use multi-platform-builder
```

```bash
docker buildx rm multi-platform-builder
```

To build and push multi-platform docker images:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t lovindata/openisr:0.0.0 --push -f Dockerfile ..
```

To clean docker cache:

```bash
docker buildx prune -af
```
