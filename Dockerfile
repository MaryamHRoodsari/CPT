FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (leveraging Docker cache)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the application port
EXPOSE 8050

# Run the application using Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8050", "--workers=4"]
