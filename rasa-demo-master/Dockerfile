FROM python:2.7-slim

SHELL ["/bin/bash", "-c"]

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

ARG GITHUB_TOKEN

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  openssh-client \
  graphviz-dev \
  pkg-config \
  git-core \
  openssl \
  libssl-dev \
  libffi6 \
  libffi-dev \
  libpng-dev \
  curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  mkdir /app

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . /app

RUN  pip install -e .

EXPOSE 5001

CMD ["python", "demo/main.py"]
