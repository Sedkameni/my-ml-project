# Use the official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements to leverage docker cache
COPY RequirementsProj_FPA.txt .

# Install dependencies
RUN pip install --no-cache-dir -r RequirementsProj_FPA.txt

# Copy the rest of the application
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "appproj_fpa.py"]
