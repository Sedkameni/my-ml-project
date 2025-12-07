import os
import subprocess
from pathlib import Path

def create_dockerfile():
    dockerfile_content = """\
# Use the official runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port 8001
EXPOSE 8001

# Run FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    print("Dockerfile created successfully")


def create_requirements():
    requirements = """\
numpy==1.26.4
pandas
scikit-learn==1.7.2
matplotlib
seaborn
joblib
uvicorn==0.22.0
fastapi==0.95.2
pydantic==1.10.7
python-multipart==0.0.6
"""

    with open("requirements.txt", 'w') as f:
        f.write(requirements)

    print("requirements written to requirements.txt")


def build_docker_image(image_name="iris-api"):
    try:
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)
        print("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error building docker image: {e}")


def run_docker_container(image_name="iris-api", port=8001):
    try:
        subprocess.run([
            "docker", "run", "-d", "-p", f"{port}:{port}",
            "--name", "iris-container", image_name
        ], check=True)
        print("Docker container ran successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running docker container: {e}")


def main():
    create_dockerfile()
    create_requirements()
    build_docker_image()
    run_docker_container()


if __name__ == '__main__':
    main()
