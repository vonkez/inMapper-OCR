FROM python:3.11-bookworm
COPY . .
RUN pip install -r ./requirements.txt
RUN pip install opencv-contrib-python
CMD ["python", "./main.py"]
