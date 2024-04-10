# Use the official Python image from Docker Hub
FROM python:3.8-slim

RUN sudo apt-get update
RUN sudo apt-get upgrade -y
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
RUN sudo apt-get install azure-cli -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
#CMD ["python", "app.py"]
CMD ["func", "start"]
