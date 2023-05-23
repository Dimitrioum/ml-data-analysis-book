FROM qts8n/cuda-python:runtime

RUN apt-get update && apt-get install python3.7-dev ffmpeg libsm6 libxext6 libpq-dev gcc -y --no-install-recommends

MAINTAINER Dmitrii Malov

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.1/compat/

ADD web-ml /usr/src/app

WORKDIR /usr/src/app

COPY web-ml/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
