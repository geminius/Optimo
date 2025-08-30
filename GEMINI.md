# GEMINI.md - Robotics Model Optimization Platform

## Project Overview

This project is a comprehensive platform for optimizing robotics models. It is designed to be a powerful, flexible, and easy-to-use tool for robotics researchers and engineers. The platform is built with a modern architecture, using a Python/FastAPI backend and a React/TypeScript frontend. It leverages a multi-agent architecture to automate the process of model optimization, from analysis and planning to execution and evaluation.

**Key Technologies:**

*   **Backend:** Python, FastAPI, SQLAlchemy, PyTorch, Transformers, BitsAndBytes
*   **Frontend:** React, TypeScript, Ant Design, Recharts, Socket.io
*   **Database:** PostgreSQL
*   **Cache:** Redis
*   **Containerization:** Docker

**Architecture:**

The platform is composed of several services that work together to provide a seamless user experience:

*   **API Server:** The main entry point for the platform, providing a RESTful API for interacting with the platform.
*   **Web UI:** A user-friendly web interface for managing models, optimizations, and results.
*   **Optimization Workers:** Asynchronous workers that perform the actual model optimization tasks.
*   **Database:** A PostgreSQL database for storing all platform data, including models, optimizations, and results.
*   **Cache:** A Redis cache for storing frequently accessed data and for inter-service communication.

## Building and Running

The project uses Docker for containerization, so the easiest way to get started is to use the provided Docker Compose files.

**Build the Docker images:**

```bash
make build
```

**Deploy the platform:**

```bash
make deploy
```

This will start all the services in the background. You can then access the web UI at `http://localhost:3000` and the API at `http://localhost:8000`.

**Run tests:**

```bash
make test
```

## Development Conventions

**Backend:**

*   The backend is written in Python and uses the FastAPI framework.
*   The code is formatted with Black and checked with Flake8 and MyPy.
*   Tests are written with Pytest.

**Frontend:**

*   The frontend is written in TypeScript and uses the React framework.
*   The code is formatted with Prettier and checked with ESLint.
*   Tests are written with Jest and React Testing Library.

**Commits:**

*   Commit messages should follow the Conventional Commits specification.
