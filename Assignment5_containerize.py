import os
import subprocess
from pathlib import Path


def create_dockerfile():
    dockerfile_content = """\
# Use the official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements to leverage docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
"""

    with open("Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    print(" Dockerfile created successfully")


def create_requirements():
    requirements = """\
Flask==2.3.0
scikit-learn==1.7.2
numpy==1.26.4
"""

    with open("requirements.txt", 'w') as f:
        f.write(requirements)
    print("requirements.txt created successfully")


def build_docker_image(image_name="iris-logistic-api"):
    try:
        print(f"\nBuilding Docker image: {image_name}...")
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)
        print(f"Docker image '{image_name}' built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error building Docker image: {e}")
        return False
    except FileNotFoundError:
        print(" Error: Docker is not installed or not in PATH")
        return False


def stop_and_remove_container(container_name="iris-container"):
    try:
        print(f"\nChecking for existing container: {container_name}...")
        # Try to stop the container
        result = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Stopped existing container")

        # Try to remove the container
        result = subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Removed existing container")
        else:
            print(f" No existing container to remove")
    except Exception as e:
        print(f"  Note: {e}")


def run_docker_container(image_name="iris-logistic-api", port=5000):
    try:
        print(f"\nRunning Docker container on port {port}...")
        subprocess.run([
            "docker", "run", "-d",
            "-p", f"{port}:{port}",
            "--name", "iris-container",
            image_name
        ], check=True)
        print(f" Docker container 'iris-container' is running on port {port}")
        print(f" Access the API at: http://localhost:{port}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        return False
    except FileNotFoundError:
        print("Error: Docker is not installed or not in PATH")
        return False


def check_files():
    """Check if required files exist"""
    missing_files = []

    if not os.path.exists("logistic_model.pkl"):
        missing_files.append("logistic_model.pkl")

    if not os.path.exists("app.py"):
        missing_files.append("app.py")

    return missing_files


def main():
    print("=" * 70)
    print("  Flask Logistic Regression Docker Containerization")
    print("=" * 70)

    # Check if required files exist
    print("\nStep 1: Checking required files...")
    missing_files = check_files()

    if missing_files:
        print(f"\n Warning: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")

        if "logistic_model.pkl" in missing_files:
            print("\n  To create logistic_model.pkl:")
            print("Run 'python logistic_model.py' first to train the model")

        if "app.py" in missing_files:
            print("\n  To create app.py:")
            print("Ensure app.py is in the current directory")

        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nExiting...")
            return
    else:
        print(" All required files found")

    print("\nStep 2: Creating Dockerfile...")
    create_dockerfile()

    print("\nStep 3: Creating requirements.txt...")
    create_requirements()

    print("\nStep 4: Stopping any existing container...")
    stop_and_remove_container()

    print("\nStep 5: Building Docker image...")
    if not build_docker_image():
        print("\n Deployment failed at build stage")
        return

    print("\nStep 6: Running Docker container...")
    if not run_docker_container():
        print("\n Deployment failed at run stage")
        return


if __name__ == '__main__':
    main()