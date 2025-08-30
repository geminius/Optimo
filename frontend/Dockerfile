# Multi-stage build for React frontend
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code and build
COPY . .
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]