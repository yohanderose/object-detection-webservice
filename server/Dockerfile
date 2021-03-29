FROM python:3.7-slim

WORKDIR /home
RUN apt update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
	pip install --upgrade pip && \
	pip install Flask flask-restful opencv-contrib-python 

CMD ["python", "api.py"]
