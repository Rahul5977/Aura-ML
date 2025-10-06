# ML_Proj

## Branching Strategy

This project uses **GitFlow**:

- `main`: Production-ready code
- `develop`: Active development
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation
- `hotfix/*`: Emergency fixes

## Project Structure

- `aura-backend/`: FastAPI backend
- `aura-frontend/`: Frontend (placeholder)
- `ml_scripts/`: ML scripts

## Development Environment

Run all services with Docker Compose:

```sh
docker-compose up --build
```

Access FastAPI health endpoint at [http://localhost:8000/health](http://localhost:8000/health)
