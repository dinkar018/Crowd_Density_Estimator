# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies (OpenCV/PIL requirements)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# Note: Using 'requirements' as the filename based on the project structure
COPY requirements .
RUN pip install --no-cache-dir -r requirements
RUN pip install --no-cache-dir waitress

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application using the built-in loader
CMD ["python", "app.py", "--port", "5000"]
