# Robotics Model Optimization Platform - Deployment Makefile

.PHONY: help build deploy deploy-dev deploy-prod test validate clean backup logs

# Default environment
ENV ?= development

help: ## Show this help message
	@echo "Robotics Model Optimization Platform - Deployment Commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	@echo "🏗️  Building Docker images..."
	docker-compose build --no-cache

deploy: ## Deploy platform (default: development)
	@echo "🚀 Deploying platform ($(ENV))..."
	./scripts/deploy.sh $(ENV)

deploy-dev: ## Deploy development environment
	@$(MAKE) deploy ENV=development

deploy-prod: ## Deploy production environment
	@$(MAKE) deploy ENV=production

test: ## Run deployment tests
	@echo "🧪 Running deployment tests..."
	./scripts/test-deployment.sh $(ENV)

validate: ## Validate deployment
	@echo "🔍 Validating deployment..."
	./scripts/validate-deployment.sh $(ENV)

clean: ## Clean up containers and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f

backup: ## Create database backup
	@echo "💾 Creating backup..."
	./scripts/backup.sh

logs: ## Show service logs
	@echo "📋 Showing logs..."
	docker-compose logs -f

logs-api: ## Show API logs
	docker-compose logs -f api

logs-worker: ## Show worker logs
	docker-compose logs -f worker

logs-db: ## Show database logs
	docker-compose logs -f db

status: ## Show service status
	@echo "📊 Service Status:"
	docker-compose ps
	@echo ""
	@echo "📈 Resource Usage:"
	docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

restart: ## Restart all services
	@echo "🔄 Restarting services..."
	docker-compose restart

restart-api: ## Restart API service
	docker-compose restart api

restart-worker: ## Restart worker service
	docker-compose restart worker

scale-workers: ## Scale worker services (use WORKERS=n)
	@echo "📈 Scaling workers to $(WORKERS)..."
	docker-compose up -d --scale worker=$(WORKERS)

update: ## Update and redeploy
	@echo "🔄 Updating deployment..."
	git pull
	docker-compose pull
	docker-compose up -d --build

setup-env: ## Setup environment file
	@if [ ! -f .env.$(ENV) ]; then \
		echo "📝 Creating .env.$(ENV) from template..."; \
		cp .env.example .env.$(ENV); \
		echo "✏️  Please edit .env.$(ENV) with your configuration"; \
	else \
		echo "✅ .env.$(ENV) already exists"; \
	fi

init: setup-env ## Initialize project for first deployment
	@echo "🎯 Initializing project..."
	mkdir -p uploads test_results backups nginx/ssl
	@echo "✅ Project initialized"
	@echo "📝 Next steps:"
	@echo "  1. Edit .env.$(ENV) with your configuration"
	@echo "  2. Run 'make deploy' to start the platform"

# Development helpers
dev-shell: ## Open shell in API container
	docker-compose exec api bash

db-shell: ## Open database shell
	docker-compose exec db psql -U postgres robotics_optimization

redis-shell: ## Open Redis shell
	docker-compose exec redis redis-cli

# Monitoring
monitor: ## Show real-time resource usage
	watch -n 2 'docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"'

health: ## Check service health
	@echo "🏥 Health Check Results:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API health check failed"
	@curl -s http://localhost:3000 > /dev/null && echo "✅ Frontend accessible" || echo "❌ Frontend not accessible"
	@docker-compose exec -T db pg_isready -U postgres && echo "✅ Database ready" || echo "❌ Database not ready"
	@docker-compose exec -T redis redis-cli ping > /dev/null && echo "✅ Redis ready" || echo "❌ Redis not ready"