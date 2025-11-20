# Aura ML - Production Folder Structure Plan

## New Structure

```
Aura-ML/
├── aura_ml/                          # Main Python package
│   ├── __init__.py
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py              # Global settings
│   │   └── model_config.py          # Model configurations
│   │
│   ├── models/                       # Model definitions and utilities
│   │   ├── __init__.py
│   │   ├── ece_classifier.py        # ECE emotion classifier
│   │   └── llm_wrapper.py           # LLM model wrapper
│   │
│   ├── training/                     # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training logic
│   │   ├── data_loader.py           # Dataset loading
│   │   └── callbacks.py             # Training callbacks
│   │
│   ├── inference/                    # Inference modules
│   │   ├── __init__.py
│   │   ├── chatbot.py               # Chatbot implementation
│   │   └── batch_processor.py       # Batch inference
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── preprocessor.py          # Data preprocessing
│   │   └── dataset_builder.py       # Dataset construction
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── logging.py               # Logging utilities
│       ├── metrics.py               # Evaluation metrics
│       └── visualization.py         # Plotting utilities
│
├── scripts/                          # Executable scripts
│   ├── train_ece.py                 # Train ECE classifier
│   ├── train_llm.py                 # Train LLM
│   ├── generate_prompts.py          # Generate training prompts
│   ├── evaluate_model.py            # Model evaluation
│   └── deploy_model.py              # Model deployment
│
├── api/                              # API/Backend (FastAPI)
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── routers/                     # API routers
│   │   ├── __init__.py
│   │   ├── chat.py                  # Chat endpoints
│   │   ├── emotion.py               # Emotion detection endpoints
│   │   └── health.py                # Health check
│   ├── models/                      # API data models
│   │   ├── __init__.py
│   │   ├── request.py               # Request schemas
│   │   └── response.py              # Response schemas
│   └── services/                    # Business logic
│       ├── __init__.py
│       ├── chat_service.py          # Chat service
│       └── emotion_service.py       # Emotion detection service
│
├── cli/                              # Command-line interface
│   ├── __init__.py
│   ├── chat.py                      # Interactive chat CLI
│   ├── train.py                     # Training CLI
│   └── utils.py                     # CLI utilities
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_ece_model.py
│   │   ├── test_llm_wrapper.py
│   │   └── test_data_loader.py
│   ├── integration/                 # Integration tests
│   │   ├── test_training.py
│   │   └── test_inference.py
│   └── fixtures/                    # Test fixtures
│       └── sample_data.json
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_ece_training.ipynb
│   └── 03_llm_finetuning.ipynb
│
├── data/                             # Data directory
│   ├── raw/                         # Raw datasets
│   ├── processed/                   # Processed datasets
│   ├── models/                      # Saved models
│   │   ├── ece/                     # ECE models
│   │   └── llm/                     # LLM models
│   └── outputs/                     # Training outputs
│
├── docs/                             # Documentation
│   ├── API.md                       # API documentation
│   ├── TRAINING.md                  # Training guide
│   ├── INFERENCE.md                 # Inference guide
│   └── DEPLOYMENT.md                # Deployment guide
│
├── configs/                          # Configuration files
│   ├── training/                    # Training configs
│   │   ├── ece_config.yaml
│   │   └── llm_config.yaml
│   └── deployment/                  # Deployment configs
│       ├── docker-compose.yml
│       └── nginx.conf
│
├── docker/                           # Docker files
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   └── .dockerignore
│
├── .github/                          # GitHub workflows
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── requirements/                     # Requirements files
│   ├── base.txt                     # Base requirements
│   ├── training.txt                 # Training requirements
│   ├── api.txt                      # API requirements
│   └── dev.txt                      # Development requirements
│
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore file
├── setup.py                          # Package setup
├── pyproject.toml                    # Modern Python project config
├── README.md                         # Main README
└── LICENSE                           # License file
```

## Migration Plan

1. Create new structure
2. Move and refactor existing code
3. Update imports
4. Create package init files
5. Add CLI entry points
6. Create API layer
7. Update documentation
8. Add tests
