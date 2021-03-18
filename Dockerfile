FROM ubuntu:18.04
MAINTAINER  Elvis MBONING "elvis.mboningtchiaze@orange.com"
WORKDIR .
ENV FLASK_APP=middleware.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV LC_ALL=C.UTF-8 
ENV LANG=C.UTF-8
RUN apt-get update -y 
RUN apt-get -y install locales && \
    apt-get install -y python3.6 python3-pip python3-dev &&\
    apt-get install -y --upgrade gcc musl musl-dev g++
RUN sed -i '/en_FR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
COPY . .
RUN pip3 install pipenv setuptools_rust
RUN pip3 install -r requirements.txt
RUN python3 -m snips_nlu download-all-languages && \
    python3 -m snips_nlu download-language-entities en && \
    python3 -m snips_nlu download-language-entities fr && \
    python3 -m snips_nlu download-language-entities de && \
    python3 -m snips_nlu download-language-entities es && \
    python3 -m snips_nlu download-language-entities it
EXPOSE 5000
CMD ["gunicorn", "-b 0.0.0.0:5000", "middleware:snips_app"]