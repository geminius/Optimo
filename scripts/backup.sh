#!/bin/bash

# Database backup script for production
set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="robotics_optimization_backup_${TIMESTAMP}.sql"

echo "🗄️  Creating database backup..."

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create database backup
docker-compose exec -T db pg_dump -U postgres robotics_optimization > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

echo "✅ Backup created: ${BACKUP_DIR}/${BACKUP_FILE}.gz"

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "🧹 Old backups cleaned up"