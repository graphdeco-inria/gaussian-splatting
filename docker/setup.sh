#!/usr/bin/env bash
# This script builds and runs a docker image for local use.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
REPOSITORY=`pwd`

cd "$DIR"

#https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"


sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo docker run hello-world


#Make sure docker group is ok
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 

exit 0
