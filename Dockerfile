FROM python:3.11-bookworm
COPY . .
RUN pip install opencv-contrib-python
RUN pip install -r ./requirements.txt
CMD ["python", "./main.py"]
