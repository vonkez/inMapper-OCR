FROM python:3.11-bookworm
COPY . .
RUN apt-get update && apt-get --yes --force-yes install libgl1
RUN apt-get install --yes python3-opencv
RUN pip install -r ./requirements.txt
RUN pip install opencv-contrib-python
CMD ["python", "./main.py"]
