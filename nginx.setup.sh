# Increase local port range to allow nginx to use many ports for connections
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

sudo cp nginx/nginx.conf /etc/nginx/nginx.conf

# Reload Nginx
sudo systemctl reload nginx