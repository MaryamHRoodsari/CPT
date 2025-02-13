FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .

# Install dependencies using no cache (to avoid stale wheels)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Force reinstallation of numpy and pandas to ensure binary compatibility
RUN pip install --no-cache-dir --force-reinstall numpy pandas

COPY . .

EXPOSE 8050
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--timeout", "120"]
