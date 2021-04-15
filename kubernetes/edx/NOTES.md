minikube start
minikube status
minikube stop

minikube start --container-runtime cri-o
minikube ssh
runc list # empty for me, seems cri-o isn't used

kubectl config view
kubectl cluster-info
minikube dashboard 

kubectl proxy

To note: pods, ReplicaSets, Deployment

kubectl create deployment mynginx --image=nginx:1.15-alpine
kubectl get deploy,rs,po -l app=mynginx
kubectl scale deploy mynginx --replicas=3
kubectl describe deploy mynginx
kubectl rollout history deploy mynginx
kubectl rollout history deploy mynginx --revision=1
kubectl set image deployment mynginx nginx=nginx:1.16-alpine
kubectl rollout undo deployment mynginx --to-revision=1

Initial replicaset:
replicaset.apps/mynginx-847485c545   3         3         3       3m42s
    Image:        nginx:1.15-alpine
pod-template-hash=5bd48c6b9d
  Containers:
   nginx:
    Image:	nginx:1.16-alpine


