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

Section 2.1.8

```
docker tag kubia olpa/kubia
docker images | head
docker push olpa/kubia
docker run -p 8080:8080 -d olpa/kubia
```

Section 2.2

```
minikube start
kubectl cluster-info
minikube ssh
```

## Section 2.2.2

Using [eksctl](https://github.com/weaveworks/eksctl/) instead of `kubectl`.

```
export AWS_PROFILE=eksctl
eksctl create cluster --name=kubia --nodes=4 --region eu-central-1 --node-type=t2.small
```

