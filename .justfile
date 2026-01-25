default:
    @just --list --unsorted

# Build docker image `deepmosa-runner`
build:
    DOCKER_BUILDKIT=1 docker build -t deepmosa-runner -f docker/Dockerfile .
