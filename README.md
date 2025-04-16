# Thrasio Snowflake to BQ, AI/ML, and Looker Project

## Project Summary
Thrasio has engaged 66degrees to modernize and consolidate their technology by
reducing operational expenses related to legacy systems, optimizing data storage, and
minimizing licensing fees. Consolidate existing technical debt into a scalable and secure
modern platform using BigQuery, modernize data and analytics capabilities by
implementing a cloud-based reporting solution using Looker and integrate AI/ML
functionalities to drive value through analytics and empower end-users with self-service
reporting in Looker by developing pre-built dashboards.

Objectives for this project include:
* Reduce operational expenses by modernizing systems and optimizing data storage.
* Consolidate technical debt into a scalable platform using BigQuery and modernize
analytics with Looker.
* Integrate AI/ML and empower users with self-service reporting via Looker BI
platform.

## Repository Structure
**TODO**: The diagram below will be updated as the code base is built out.

```
├── .github/
│   └── workflows/
│       ├── deploy-application.yml # Deploy application
│       └── ruff.yml  # Ruff check configuration
├── app/  # Application
│   ├── api/
│   │   └── routes/
│   │       ├──
│   │       └──
│   ├── core/
│   │   ├── 
│   │   └── 
│   ├── models
│   │   ├── 
│   │   ├── 
│   │   ├── 
│   │   └── 
│   ├── services/
│   │   ├── 
│   │   ├── 
│   │   ├── 
│   │   └── 
│   └── app.py
├── experiments/  # Experiments
│   └── 
├── .env.example  # Example environment file
├── .gitignore  # Files and directories ignored by version control
├── poetry.lock  # Dependencies
├── pyproject.toml  # Build system requirements
└── README.md   # Project overview and instructions
```

## Environment Setup
*This project utilizes a Poetry virtual environment.*

To set up the environment, complete the following steps after cloning this GitHub repository:

**1. Install Poetry (if you have not already).**
```
brew install poetry
```
**2. Create the virtual environment with the necessary packages and dependencies.**
```
poetry install
```
**3. Initalize the environment.**
```
poetry shell
```

Additionally, you will need a .env file and a service account key:

**1. Create a .env file (following the specifications in .env.example)** and place it in the project's
root directory.

**2. Obtain a GCP service account key** and place it in the project's root directory. The file should 
be named key.json.
