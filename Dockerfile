FROM python:3.10-slim

WORKDIR /app

# Copy all application files and the models folder into the container
COPY . .
COPY models /app/models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["python", "app.py"]