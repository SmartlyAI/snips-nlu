FROM docker.elastic.co/beats/filebeat:7.6.0

ARG ELK_LOGS_HOST

# Copy our custom configurations file
COPY filebeat.yml /usr/share/filebeat/filebeat.yml

USER root

RUN sed -i "s/ELK_LOGS_HOST/$ELK_LOGS_HOST/g" /usr/share/filebeat/filebeat.yml

# Create a directory to map volume with all docker log files
RUN mkdir /usr/share/filebeat/dockerlogs
RUN chown -R root /usr/share/filebeat/
RUN chmod -R go-w /usr/share/filebeat/