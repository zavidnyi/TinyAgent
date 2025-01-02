FROM --platform=linux/amd64 python:3.10-slim

RUN groupadd -g 1024 trainer && useradd -u 1024 -m trainer -g 1024
USER 1024

COPY --chown=trainer:trainer requirements.txt /home/trainer/
WORKDIR /home/trainer/

RUN pip install torch==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

COPY --chown=trainer:trainer . /home/trainer/