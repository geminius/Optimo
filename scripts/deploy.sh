#!/bin/bash

# Robotics Model Optimization Platform Deployment Script
set -e

# Configuration
ENVIRONMENT=${1:-development}
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.${ENVIRONMENT}"

echo "🚀 Deploying Robotics Model Optimization Platform (${ENVIRONMENT})"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file $ENV_FILE not found!"
    echo "Please create it based on .env.example"
    exit 1
fi

# Load environment variables
export $(cat $ENV_FILE | grep -v '^#' | xargs)

# Determine compose files based on environment
case $ENVIRONMENT in
    "development")
        COMPOSE_FILES="-f $COMPOSE_FILE -f docker-compose.dev.yml"
        ;;
    "production")
        COMPOSE_FILES="-f $COMPOSE_FILE -f docker-compose.prod.yml"
        ;;
    *)
        COMPOSE_FILES="-f $COMPOSE_FILE"
        ;;
esac

echo "📋 Using compose files: $COMPOSE_FILES"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed!"
    exit 1
fi

# Check required directories
mkdir -p uploads test_results backups nginx/ssl

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose $COMPOSE_FILES build --no-cache
docker-compose $COMPOSE_FILES up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
timeout=300
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if docker-compose $COMPOSE_FILES ps | grep -q "Up (healthy)"; then
        echo "✅ Services are healthy!"
        break
    fi
    
    echo "⏳ Waiting for services... ($elapsed/$timeout seconds)"
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ $elapsed -ge $timeout ]; then
    echo "❌ Services failed to become healthy within $timeout seconds"
    echo "📋 Service status:"
    docker-compose $COMPOSE_FILES ps
    echo "📋 Logs:"
    docker-compose $COMPOSE_FILES logs --tail=50
    exit 1
fi

# Run database migrations if needed
echo "🗄️  Running database migrations..."
docker-compose $COMPOSE_FILES exec -T api python -c "
from src.models.store import ModelStore
store = ModelStore()
store.initialize_database()
print('Database initialized successfully')
"

# Display deployment information
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service Status:"
docker-compose $COMPOSE_FILES ps

echo ""
echo "🌐 Access URLs:"
if [ "$ENVIRONMENT" = "production" ]; then
    echo "  Frontend: https://your-domain.com"
    echo "  API: https://your-domain.com/api"
else
    echo "  Frontend: http://localhost:3000"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
fi

echo ""
echo "📋 Useful Commands:"
echo "  View logs: docker-compose $COMPOSE_FILES logs -f"
echo "  Stop services: docker-compose $COMPOSE_FILES down"
echo "  Restart services: docker-compose $COMPOSE_FILES restart"
echo "  Scale workers: docker-compose $COMPOSE_FILES up -d --scale worker=4"