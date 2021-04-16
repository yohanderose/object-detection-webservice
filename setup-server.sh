# Install Kubectl
sudo apt update && sudo apt install -y snap && sudo snap install kubectl --classicgit clone https://github.com/yohanderose/object-detection-webservice.git
# Install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.10.0/kind-linux-amd64
chmod +x ./kind
sudo mv kind /usr/bin
# Fetch repo
git clone https://github.com/yohanderose/object-detection-webservice.git
cd object-detection-webservice/
# Start docker deamon
sudo systemctl start docker
# Create Cluster and spin up Pods
sudo kind create cluster --config kind-config.yml && sudo kubectl apply -f deployment.yml
