# Deployment Guide

This guide covers deploying the Robotics Model Optimization Platform using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 50GB+ disk space
- (Optional) NVIDIA GPU with CUDA support for optimization acceleration

## Quick Start

### Development Deployment

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd robotics-model-optimization-platform
   cp .env.example .env.development
   ```

2. **Edit environment variables**:
   ```bash
   nano .env.development
   # Update database passwords and other settings
   ```

3. **Deploy**:
   ```bash
   ./scripts/deploy.sh development
   ```

4. **Access the platform**:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Production Deployment

1. **Setup environment**:
   ```bash
   cp .env.example .env.production
   nano .env.production
   # Set secure passwords and production URLs
   ```

2. **Configure SSL certificates**:
   ```bash
   mkdir -p nginx/ssl
   # Copy your SSL certificates to nginx/ssl/cert.pem and nginx/ssl/key.pem
   ```

3. **Deploy**:
   ```bash
   ./scripts/deploy.sh production
   ```

## Architecture Overview

The platform consists of the following services:

- **API**: FastAPI backend service
- **Worker**: Background optimization processing
- **Frontend**: React web interface
- **Database**: PostgreSQL for persistent storage
- **Redis**: Caching and message queuing
- **Nginx**: Reverse proxy and load balancer

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | Database name | `robotics_optimization` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | Required |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379` |
| `SECRET_KEY` | API secret key | Required |
| `JWT_SECRET` | JWT signing secret | Required |
| `MAX_CONCURRENT_OPTIMIZATIONS` | Worker limit | `5` |
| `OPTIMIZATION_TIMEOUT` | Timeout in seconds | `3600` |

### Service Configuration

#### API Service
- Handles REST API requests
- Manages WebSocket connections
- Coordinates optimization workflows

#### Worker Service
- Processes optimization tasks
- Scales horizontally for increased throughput
- Supports GPU acceleration when available

#### Database Service
- PostgreSQL with automatic backups
- Persistent volume for data storage
- Health checks and recovery

## Scaling

### Horizontal Scaling

Scale workers for increased optimization throughput:
```bash
docker-compose up -d --scale worker=4
```

Scale API instances for higher request volume:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=3
```

### Resource Limits

Production deployment includes resource limits:
- API: 2 CPU cores, 4GB RAM
- Worker: 4 CPU cores, 8GB RAM
- Database: 2 CPU cores, 4GB RAM

## Monitoring

### Health Checks

All services include health checks:
- API: HTTP endpoint `/health`
- Database: PostgreSQL connection test
- Redis: Redis ping command
- Workers: Python import test

### Logs

View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
```

### Metrics

The platform exposes metrics for monitoring:
- Optimization success/failure rates
- Processing times
- Resource utilization
- Queue depths

## Backup and Recovery

### Database Backup

Automated backup script:
```bash
./scripts/backup.sh
```

Manual backup:
```bash
docker-compose exec db pg_dump -U postgres robotics_optimization > backup.sql
```

### Restore from Backup

```bash
docker-compose exec -T db psql -U postgres robotics_optimization < backup.sql
```

### Volume Backup

Backup persistent volumes:
```bash
docker run --rm -v robotics_optimization_postgres_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
```

## Security

### Production Security Checklist

- [ ] Change default passwords
- [ ] Configure SSL certificates
- [ ] Set secure JWT secrets
- [ ] Enable CORS restrictions
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Regular security updates

### Network Security

- Services communicate over internal Docker network
- Only necessary ports exposed to host
- Nginx provides SSL termination and rate limiting

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs

# Restart services
docker-compose restart
```

#### Database Connection Issues
```bash
# Check database health
docker-compose exec db pg_isready -U postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

#### Out of Memory
```bash
# Check resource usage
docker stats

# Reduce worker count
docker-compose up -d --scale worker=1
```

#### GPU Not Available
```bash
# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Update docker-compose for GPU support
# Add to worker service:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

### Performance Tuning

#### Database Performance
- Increase shared_buffers for large datasets
- Configure connection pooling
- Regular VACUUM and ANALYZE

#### Worker Performance
- Adjust worker count based on CPU cores
- Monitor memory usage during optimizations
- Use GPU acceleration when available

#### API Performance
- Enable response caching
- Configure connection pooling
- Use CDN for static assets

## Maintenance

### Regular Tasks

1. **Database maintenance**:
   ```bash
   docker-compose exec db psql -U postgres -c "VACUUM ANALYZE;"
   ```

2. **Log rotation**:
   ```bash
   docker system prune -f
   ```

3. **Security updates**:
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

### Monitoring Checklist

- [ ] Service health status
- [ ] Disk space usage
- [ ] Memory consumption
- [ ] Optimization queue depth
- [ ] Error rates
- [ ] Response times

## Support

For deployment issues:
1. Check this documentation
2. Review service logs
3. Verify environment configuration
4. Check resource availability
5. Contact support with logs and configuration details