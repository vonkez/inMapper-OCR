FROM python:3.11-bookworm
COPY . .
RUN apt-get update && apt-get install libgl1
RUN pip install -r ./requirements.txt
RUN pip install opencv-contrib-python
CMD ["python", "./main.py"]
