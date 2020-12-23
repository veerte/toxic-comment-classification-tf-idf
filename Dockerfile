FROM python:3.9

WORKDIR /app
COPY Pipfile Pipfile.lock ./

RUN pip install pipenv
RUN pipenv install --deploy



