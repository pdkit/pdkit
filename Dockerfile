# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: George Roussos <george@roussos.mobi>

FROM python:3

WORKDIR /home/pdkit

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pdkit
RUN pip install --no-cache-dir jupyter

COPY . .
COPY notebooks/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

WORKDIR /home/pdkit/notebooks

RUN mkdir data
RUN mkdir extra
VOLUME /home/pdkit/notebooks/data
VOLUME /home/pdkit/notebooks/extra

CMD ["/usr/local/bin/jupyter-notebook"]
