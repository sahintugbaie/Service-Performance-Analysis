# Beta Regression & Clustering Analysis Tool

## Overview

This is a web-based statistical analysis application built with Streamlit that performs beta regression and clustering analysis on Excel data. The application provides an interactive interface for users to upload Excel files, select variables for analysis, and generate comprehensive visualizations and statistical summaries.

The tool is designed for researchers and analysts who need to perform advanced statistical modeling on proportion data (beta regression) and identify patterns through clustering algorithms.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

**Frontend**: Streamlit web interface providing interactive widgets and real-time visualization
**Backend**: Python-based statistical computing engine with modular components
**Data Processing**: Pandas-based data manipulation and validation
**Statistical Analysis**: Custom implementations of beta regression and clustering algorithms
**Visualization**: Matplotlib/Seaborn-based plotting with Streamlit integration

The architecture prioritizes ease of use while maintaining statistical rigor and computational efficiency.

## Key Components

### Core Modules

1. **app.py** - Main Streamlit application entry point
   - Manages session state for data persistence
   - Orchestrates user interface components
   - Handles file uploads and user interactions

2. **data_processing.py** - Data handling and validation
   - Excel file loading with openpyxl engine
   - Data cleaning and preprocessing
   - Statistical validation for beta regression requirements

3. **r_scripts.py** - Statistical analysis engine
   - Beta regression implementation using logit transformation
   - Multiple clustering algorithms (K-means, hierarchical)
   - Statistical model fitting and evaluation

4. **visualization.py** - Plotting and chart generation
   - Correlation matrices and heatmaps
   - Distribution plots and histograms
   - Regression diagnostics and residual plots
   - Clustering visualizations

### Session State Management

The application maintains persistent state across user interactions:
- Uploaded dataset
- Analysis results (beta regression, clustering)
- Selected variables and parameters
- Visualization preferences

## Data Flow

1. **Data Ingestion**: User uploads Excel file → data_processing validates and loads data
2. **Variable Selection**: User selects dependent/independent variables through Streamlit interface
3. **Analysis Execution**: Statistical models run in r_scripts module
4. **Results Storage**: Results cached in Streamlit session state
5. **Visualization**: Charts generated dynamically based on analysis results
6. **Interactive Exploration**: Users can modify parameters and re-run analyses

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **scipy**: Scientific computing and clustering
- **sklearn**: Machine learning algorithms
- **statsmodels**: Statistical modeling
- **openpyxl**: Excel file processing

### File Dependencies
- Excel files (.xlsx format) for data input
- No external databases or APIs required

## Deployment Strategy

The application is designed for single-user deployment scenarios:

**Local Development**: Run directly with `streamlit run app.py`
**Cloud Deployment**: Compatible with Streamlit Cloud, Heroku, or similar platforms
**Containerization**: Can be containerized with Docker for consistent deployment

Key considerations:
- All dependencies installable via pip
- No database setup required
- Stateless design (session state only)
- File-based data input only

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- July 01, 2025. Initial setup