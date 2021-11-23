#!/bin/bash -x

docker build -t cyclegan_streamlit:prod .
docker save cyclegan_streamlit:prod | gzip > cyclegan_streamlit_image.tar.gz
rsync -aP react_web.tar.gz "$1":/home/gs/station_agent_web

rm cyclegan_streamlit_image.tar.gz

ssh "$1" << EOF
  cd cyclegan_streamlit
  docker image rm cyclegan_streamlit:prod
  docker load --input cyclegan_streamlit_image.tar.gz
  docker-compose down
  docker-compose up -d
  docker rmi $(docker images -f "dangling=true" "until=24h" -q)
EOF

