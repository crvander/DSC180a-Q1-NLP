ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

COPY . ./

CMD ["python", "-r", "install", "requirements.txt"]
CMD ["python", "run.py"]
CMD ["/bin/bash"]