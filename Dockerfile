FROM python:3.8
RUN apt-get update && apt-get install -y python3.8-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run frontend/cyclegan_streamlit.py --server.port 80 --server.headless true --server.runOnSave true
EXPOSE 80