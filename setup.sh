# Install NFS protcol for snips-data
sudo apt-get install nfs-kernel-server -y
sudo chown -R $USER:$USER /snips-data
echo "/snips-data    ${DISTANT_SERVER_PRIVATE_IP}(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo systemctl restart nfs-kernel-server

# Increase local port range to allow nginx to use many ports for connections
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Init docker swarm
docker swarm init

# Create registry which will contains all images of smartly services
docker service create --name registry-smartly-swarm --publish 5000:5000 --constraint 'node.role == manager' --detach=false registry:2

# Create volumes
docker volume create --name=rabbitmq

# Launch the stack named snipsnlu
env $(cat .env | grep ^[A-Z] | xargs) docker stack deploy --compose-file docker-compose.yml snipsnlu
