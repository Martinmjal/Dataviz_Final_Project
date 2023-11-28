FROM python:3.11.5

# Set a working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy your app and all other files
COPY ./index.py /app/index.py
COPY ./*.csv /app/
COPY ./*.pkl /app/

# Run the app
CMD ["python", "/app/index.py"]