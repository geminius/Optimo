#!/bin/bash

# Comprehensive deployment testing script
set -e

ENVIRONMENT=${1:-development}
CLEANUP=${2:-false}

echo "🚀 Testing deployment for environment: $ENVIRONMENT"

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = "true" ]; then
        echo "🧹 Cleaning up test deployment..."
        docker-compose down -v
        docker system prune -f
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Deploy the platform
echo "📦 Deploying platform..."
./scripts/deploy.sh $ENVIRONMENT

# Wait for services to stabilize
echo "⏳ Waiting for services to stabilize..."
sleep 60

# Validate deployment
echo "🔍 Validating deployment..."
./scripts/validate-deployment.sh $ENVIRONMENT

# Run integration tests
echo "🧪 Running integration tests..."
python -m pytest tests/integration/ -v --tb=short

# Run performance tests
echo "⚡ Running performance tests..."
python -m pytest tests/performance/ -v --tb=short

# Test backup functionality
echo "💾 Testing backup functionality..."
./scripts/backup.sh

# Verify backup was created
if [ -f "backups/robotics_optimization_backup_*.sql.gz" ]; then
    echo "✅ Backup created successfully"
else
    echo "❌ Backup creation failed"
    exit 1
fi

# Test scaling
echo "📈 Testing service scaling..."
docker-compose up -d --scale worker=3
sleep 30

# Verify scaled services
WORKER_COUNT=$(docker-compose ps worker | grep -c "Up")
if [ $WORKER_COUNT -eq 3 ]; then
    echo "✅ Service scaling successful"
else
    echo "❌ Service scaling failed"
    exit 1
fi

# Scale back down
docker-compose up -d --scale worker=1

echo "🎉 All deployment tests passed successfully!"

# Display final status
echo ""
echo "📊 Final Status:"
docker-compose ps
echo ""
echo "💡 Deployment is ready for use!"