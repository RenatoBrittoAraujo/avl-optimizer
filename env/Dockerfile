FROM matthewfeickert/docker-python3-ubuntu
COPY ./ /app

RUN sudo apt-get update
RUN sudo apt-get upgrade -y
# RUN apt-get install python3

WORKDIR /app

CMD ["python3", "avl_optmizer.py"]
