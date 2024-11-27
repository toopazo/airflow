
FROM python

WORKDIR /code
COPY ../requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# COPY ${CODE_PATH} /code/app
# CMD ["fastapi", "run", "app/main.py", "--port", "80"]
CMD bash