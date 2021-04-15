Chapter 2: First steps with Docker and Kubernetes

```
docker build -t kubia .
docker images
docker image save kubia >x.tar
docker run --name kubia-container -p 8080:8080 -d kubia
curl http://localhost:8080
docker ps
docker inspect kubia-container
docker exec -it kubia-container bash
ps aux
ps aux | grep app.js
docker stop kubia-container
docker rm kubia-container
```
