mkdir -p ~/.docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSEV2_VERSION_V2.0.0-rc.3}/docker-compose-linux-amd64" -o ~/.docker/cli-plugins/docker-compose
sudo chmod +x ~/.docker/cli-plugins/docker-compose
echo "docker compose installed"
