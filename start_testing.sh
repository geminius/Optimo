#!/bin/bash

# Robotics Model Optimization Platform - Testing Startup Script
# This script helps you quickly start the API server and run tests

set -e

echo "=========================================="
echo "🤖 Robotics Model Optimization Platform"
echo "   Testing Startup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Check if dependencies are installed
print_info "Checking dependencies..."
if ! python3 -c "import fastapi" &> /dev/null; then
    print_warning "Dependencies not installed. Installing..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Show menu
echo ""
echo "=========================================="
echo "What would you like to do?"
echo "=========================================="
echo "1) Start API Server"
echo "2) Run Automated Test Script"
echo "3) Start API Server + Run Test"
echo "4) Start Frontend (in new terminal)"
echo "5) Run Integration Tests"
echo "6) View API Documentation"
echo "7) Exit"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        print_info "Starting API Server..."
        echo ""
        print_success "API Server will be available at:"
        echo "   - API: http://localhost:8000"
        echo "   - Docs: http://localhost:8000/docs"
        echo "   - Health: http://localhost:8000/health"
        echo ""
        print_info "Press Ctrl+C to stop the server"
        echo ""
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    
    2)
        print_info "Running automated test script..."
        echo ""
        print_warning "Make sure API server is running in another terminal!"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_real_model.py
        ;;
    
    3)
        print_info "Starting API Server in background..."
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &
        API_PID=$!
        print_success "API Server started (PID: $API_PID)"
        
        # Wait for server to start
        print_info "Waiting for API server to be ready..."
        sleep 5
        
        # Check if server is running
        if curl -s http://localhost:8000/health > /dev/null; then
            print_success "API Server is ready!"
            echo ""
            print_info "Running automated test script..."
            python test_real_model.py
            
            # Ask if user wants to keep server running
            echo ""
            read -p "Keep API server running? (y/n): " keep_running
            if [ "$keep_running" != "y" ]; then
                print_info "Stopping API server..."
                kill $API_PID
                print_success "API server stopped"
            else
                print_success "API server still running (PID: $API_PID)"
                print_info "To stop it later, run: kill $API_PID"
            fi
        else
            print_error "API Server failed to start. Check api_server.log for details"
            kill $API_PID 2>/dev/null || true
        fi
        ;;
    
    4)
        print_info "Starting Frontend..."
        echo ""
        if [ ! -d "frontend/node_modules" ]; then
            print_warning "Node modules not installed. Installing..."
            cd frontend
            npm install
            cd ..
            print_success "Node modules installed"
        fi
        
        print_success "Frontend will be available at: http://localhost:3000"
        echo ""
        print_info "Press Ctrl+C to stop the frontend"
        echo ""
        cd frontend
        npm start
        ;;
    
    5)
        print_info "Running integration tests..."
        echo ""
        pytest tests/integration/test_real_optimization_workflow.py -v -s
        ;;
    
    6)
        print_info "Opening API documentation..."
        echo ""
        print_success "API Documentation URLs:"
        echo "   - Swagger UI: http://localhost:8000/docs"
        echo "   - ReDoc: http://localhost:8000/redoc"
        echo ""
        print_warning "Make sure API server is running!"
        echo ""
        
        # Try to open in browser (macOS)
        if command -v open &> /dev/null; then
            open http://localhost:8000/docs
        else
            print_info "Please open http://localhost:8000/docs in your browser"
        fi
        ;;
    
    7)
        print_info "Exiting..."
        exit 0
        ;;
    
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Done!"
