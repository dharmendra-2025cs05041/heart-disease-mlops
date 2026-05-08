#!/bin/bash

# Deployment Script for Heart Disease Prediction API
# Supports Docker and Kubernetes deployment

set -e

DEPLOYMENT_TYPE=${1:-docker}
IMAGE_NAME="heart-disease-api"
IMAGE_TAG="latest"

echo "======================================"
echo "Heart Disease API Deployment"
echo "======================================"
echo ""

if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
    echo "Deployment Type: Docker"
    echo ""
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME:$IMAGE_TAG -f deployment/docker/Dockerfile .
    echo "✓ Docker image built successfully"
    echo ""
    
    # Stop existing container if running
    echo "Stopping existing container (if any)..."
    docker stop $IMAGE_NAME 2>/dev/null || true
    docker rm $IMAGE_NAME 2>/dev/null || true
    echo ""
    
    # Run container
    echo "Starting container..."
    docker run -d \
        -p 8000:8000 \
        --name $IMAGE_NAME \
        $IMAGE_NAME:$IMAGE_TAG
    echo "✓ Container started successfully"
    echo ""
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    sleep 5
    
    # Test API
    echo "Testing API health..."
    curl -f http://localhost:8000/health || echo "Health check failed"
    echo ""
    echo ""
    
    echo "======================================"
    echo "Deployment Completed!"
    echo "======================================"
    echo ""
    echo "API URL: http://localhost:8000"
    echo "Swagger UI: http://localhost:8000/docs"
    echo ""
    echo "View logs: docker logs $IMAGE_NAME -f"
    echo "Stop container: docker stop $IMAGE_NAME"
    
elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ] || [ "$DEPLOYMENT_TYPE" == "k8s" ]; then
    echo "Deployment Type: Kubernetes"
    echo ""
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME:$IMAGE_TAG -f deployment/docker/Dockerfile .
    echo "✓ Docker image built successfully"
    echo ""

    # Make the image visible to the cluster's container runtime.
    # Behaviour depends on which engine the cluster uses:
    #   - Docker Desktop K8s        -> kubelet shares the host dockerd, nothing to do
    #   - Rancher Desktop, CE=moby  -> k3s runs with --docker, also shares dockerd
    #   - Rancher Desktop, CE=ctrd  -> k3s uses its own containerd in the Lima VM,
    #                                  we have to pipe the image through rdctl
    #   - Minikube                  -> use `minikube image load`
    CTX=$(kubectl config current-context 2>/dev/null || echo "")
    RD_ENGINE=""
    if command -v rdctl >/dev/null 2>&1; then
        RD_ENGINE=$(rdctl list-settings 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('containerEngine',{}).get('name',''))" 2>/dev/null \
            || echo "")
    fi

    if [ "$CTX" = "rancher-desktop" ] && [ "$RD_ENGINE" = "moby" ]; then
        echo "Rancher Desktop with CE=moby — k3s shares dockerd, image already visible"
        echo ""
    elif [ "$CTX" = "rancher-desktop" ] && command -v rdctl >/dev/null 2>&1; then
        # CE=containerd path: load through the VM's k3s containerd socket.
        K3S_SOCK="/run/k3s/containerd/containerd.sock"
        if ! rdctl shell -- true >/dev/null 2>&1; then
            echo "✗ Rancher Desktop VM is not running. Start the app and re-run."
            exit 1
        fi
        echo "Loading image into Rancher Desktop k3s containerd (k8s.io namespace)..."
        docker save $IMAGE_NAME:$IMAGE_TAG \
            | rdctl shell -- sudo ctr --address "$K3S_SOCK" -n k8s.io images import -
        echo "✓ Image loaded into k3s containerd"
        echo ""
    elif command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
        echo "Loading image into Minikube..."
        minikube image load $IMAGE_NAME:$IMAGE_TAG
        echo "✓ Image loaded into Minikube"
        echo ""
    elif [ "$CTX" = "docker-desktop" ]; then
        echo "Context is docker-desktop — kubelet shares dockerd, no load step needed"
        echo ""
    else
        echo "⚠  Context '$CTX' is not Rancher Desktop, Minikube or Docker Desktop."
        echo "   Make sure the image '$IMAGE_NAME:$IMAGE_TAG' is reachable from the cluster"
        echo "   (push to a registry, or load it manually into the cluster's runtime)."
        echo ""
    fi

    # Apply Kubernetes manifests
    echo "Applying Kubernetes manifests..."
    kubectl apply -f deployment/kubernetes/deployment.yaml
    echo "✓ Kubernetes resources created"
    echo ""
    
    # Wait for deployment
    echo "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=120s deployment/heart-disease-api
    echo "✓ Deployment is ready"
    echo ""
    
    # Get service info
    echo "Service Information:"
    kubectl get services heart-disease-api-service
    echo ""
    
    # Get pods
    echo "Pod Information:"
    kubectl get pods -l app=heart-disease-api
    echo ""
    
    echo "======================================"
    echo "Deployment Completed!"
    echo "======================================"
    echo ""
    
    if [ "$CTX" = "minikube" ] && command -v minikube >/dev/null 2>&1; then
        echo "Get service URL: minikube service heart-disease-api-service --url"
    elif [ "$CTX" = "rancher-desktop" ]; then
        echo "Rancher Desktop exposes LoadBalancer services on localhost via Traefik."
        echo "Or use port forwarding: kubectl port-forward svc/heart-disease-api-service 8000:80"
    else
        echo "Get external IP: kubectl get service heart-disease-api-service"
        echo "Or use port forwarding: kubectl port-forward svc/heart-disease-api-service 8000:80"
    fi
    echo ""
    echo "View logs: kubectl logs -l app=heart-disease-api -f"
    echo "Delete deployment: kubectl delete -f deployment/kubernetes/deployment.yaml"
    
elif [ "$DEPLOYMENT_TYPE" == "monitoring" ]; then
    echo "Deployment Type: Monitoring Stack"
    echo ""
    
    # Deploy monitoring
    echo "Deploying Prometheus and Grafana..."
    kubectl apply -f deployment/kubernetes/monitoring.yaml
    echo "✓ Monitoring stack deployed"
    echo ""
    
    echo "======================================"
    echo "Monitoring Deployed!"
    echo "======================================"
    echo ""
    echo "Prometheus: http://localhost:30090"
    echo "Grafana: http://localhost:30300 (admin/admin)"
    echo ""
    echo "Access via: kubectl port-forward service/prometheus-service 9090:9090"
    echo "           kubectl port-forward service/grafana-service 3000:3000"
    
else
    echo "Invalid deployment type: $DEPLOYMENT_TYPE"
    echo ""
    echo "Usage: $0 [docker|kubernetes|monitoring]"
    echo ""
    echo "Examples:"
    echo "  $0 docker       - Deploy using Docker"
    echo "  $0 kubernetes   - Deploy to Kubernetes"
    echo "  $0 monitoring   - Deploy monitoring stack"
    exit 1
fi
