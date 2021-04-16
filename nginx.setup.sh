# Install some librairies
sudo apt-get update
sudo apt-get install -y curl python gnupg
sudo apt-get install -y nginx

# Increase local port range to allow nginx to use many ports for connections
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

sudo cp nginx/nginx.conf /etc/nginx/nginx.conf

# Install amplify
sudo curl -L -O https://github.com/nginxinc/nginx-amplify-agent/raw/master/packages/install.sh \
    && sudo API_KEY=$AMPLIFY_KEY sh ./install.sh -y    
sudo cp nginx/stub_status.conf /etc/nginx/conf.d/stub_status.conf

# Reload Nginx
sudo systemctl reload nginx