FROM python:3.11

# Install Node.js (version 18 or later) and required tools
RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first to make use of the build cache
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/
COPY graphiti ./graphiti

# Install Node and Python dependencies
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync --frozen

# Copy the project source
COPY . .

EXPOSE 3000 5001

# Start the frontend and backend together (development mode)
CMD ["npm", "run", "dev"]
