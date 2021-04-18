# Use the python slim base image
FROM python:3.7-slim 

# Set the default location in the container
WORKDIR /home

# Copy over files needed inside the container
COPY api.py api.py
COPY yolov3-tiny.cfg yolov3-tiny.cfg  
COPY yolov3-tiny.weights  yolov3-tiny.weights  
COPY coco.names coco.names

# Install dependencies
RUN apt update && \
	# Image processing deps 
    apt-get install ffmpeg libsm6 libxext6  -y && \
	pip install --upgrade pip && \
	# Webservice and Image Processing 
	pip install Flask flask-restful opencv-contrib-python 

# Start service on container build
CMD ["python", "api.py"]

# Make container accessible on this host port
EXPOSE 5009
