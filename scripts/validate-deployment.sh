#!/bin/bash

# Deployment validation script
set -e

ENVIRONMENT=${1:-development}

echo "🔍 Validating deployment for environment: $ENVIRONMENT"

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
fi

# Check if services are running
echo "📋 Checking service status..."
docker-compose ps

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run deployment validation tests
echo "🧪 Running deployment validation tests..."
python -m pytest tests/deployment/test_deployment_validation.py -v \
    --tb=short \
    --disable-warnings

# Check service health endpoints
echo "🏥 Checking service health..."

# API health check
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service health check failed"
    exit 1
fi

# Frontend accessibility check
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend accessibility check failed"
    exit 1
fi

# Database connectivity check
if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ Database is ready"
else
    echo "❌ Database connectivity check failed"
    exit 1
fi

# Redis connectivity check
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis connectivity check failed"
    exit 1
fi

# Check resource usage
echo "📊 Checking resource usage..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Check logs for errors
echo "📋 Checking for errors in logs..."
ERROR_COUNT=$(docker-compose logs --since=5m 2>&1 | grep -i error | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT errors in recent logs"
    docker-compose logs --since=5m | grep -i error | head -10
else
    echo "✅ No errors found in recent logs"
fi

echo "🎉 Deployment validation completed successfully!"

# Display useful information
echo ""
echo "🌐 Service URLs:"
echo "  Frontend: http://localhost:3000"
echo "  API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Next steps:"
echo "  1. Upload a test model via the frontend"
echo "  2. Run an optimization to verify functionality"
echo "  3. Check monitoring dashboards"
echo "  4. Review logs for any warnings"