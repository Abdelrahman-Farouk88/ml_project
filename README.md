# Student Exam Performance Predictor

An end-to-end machine learning project that predicts student math exam scores based on various demographic and academic factors. This project includes a complete ML pipeline from data ingestion to model deployment via a Flask web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Contributing](#contributing)
- [Author](#author)

## 🎯 Overview

This project predicts student math exam performance (scores out of 100) using machine learning models. The prediction is based on several input features including:

- **Gender**: Male or Female
- **Race/Ethnicity**: Group A, B, C, D, or E
- **Parental Level of Education**: Various education levels
- **Lunch Type**: Free/Reduced or Standard
- **Test Preparation Course**: None or Completed
- **Reading Score**: Score out of 100
- **Writing Score**: Score out of 100

The project follows best practices for ML engineering with a modular, scalable architecture that separates data processing, model training, and prediction pipelines.

## ✨ Features

- **Accurate Predictions**: Uses multiple ML algorithms and selects the best performing model
- **Web Interface**: User-friendly Flask web application with modern UI
- **End-to-End Pipeline**: Complete ML pipeline from data ingestion to deployment
- **Model Comparison**: Automatically tests and compares 7 different regression models
- **Data Preprocessing**: Automated feature engineering and preprocessing
- **Logging**: Comprehensive logging system for debugging and monitoring
- **Error Handling**: Custom exception handling for robust error management

## 🛠 Technologies Used

### Machine Learning & Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **catboost**: Gradient boosting on decision trees
- **xgboost**: Extreme gradient boosting
- **dill**: Object serialization

### Web Framework
- **Flask**: Web application framework

### Visualization & Analysis
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 📁 Project Structure

```
ml_project/
│
├── app.py                          # Flask application entry point
├── setup.py                        # Package setup configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── src/                            # Source code directory
│   ├── __init__.py
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Utility functions
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Feature preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   │
│   └── pipeline/                   # End-to-end pipelines
│       ├── __init__.py
│       ├── train_pipeline.py      # Training pipeline
│       └── predict_pipeline.py    # Prediction pipeline
│
├── templates/                      # HTML templates
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form page
│
├── static/                         # Static files
│   └── css/
│       └── style.css              # Stylesheet
│
├── artifacts/                      # Generated files
│   ├── data.csv                   # Raw data
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── preprocessor.pkl           # Preprocessing pipeline
│   └── model.pkl                  # Trained model
│
├── logs/                           # Application logs
├── notebook/                       # Jupyter notebooks for EDA
│   └── data/
│       └── data.csv               # Original dataset
│
└── venv/                          # Virtual environment (not in git)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd ml_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

6. **Ensure data file exists**:
   - Place your dataset at `notebook/data/data.csv`
   - The dataset should contain the required features and target variable (math score)

## 📖 Usage

### Training the Model

To train the machine learning model, run the training pipeline:

```bash
python -m src.pipeline.train_pipeline
```

This will:
1. Load and split the data into training and test sets
2. Preprocess features (encoding, scaling)
3. Train multiple models and select the best one
4. Save the trained model and preprocessor to `artifacts/`

### Running the Web Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   - Open your web browser and navigate to: `http://localhost:5000` or `http://127.0.0.1:5000`

3. **Make predictions**:
   - Click "Get Started" on the landing page
   - Fill in the student information form
   - Click "Predict Math Score" to get the prediction

### Using the Prediction Pipeline Programmatically

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create input data
data = CustomData(
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='none',
    reading_score=72,
    writing_score=74
)

# Get prediction
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(data.get_data_as_frame())
print(f"Predicted Math Score: {prediction[0]:.2f}")
```

## 🤖 Model Information

### Models Evaluated

The system automatically evaluates and compares the following regression models:

1. **Linear Regression**: Baseline linear model
2. **Gradient Boosting Regressor**: Ensemble method with gradient descent
3. **Decision Tree Regressor**: Tree-based model
4. **Random Forest Regressor**: Ensemble of decision trees
5. **XGBoost Regressor**: Optimized gradient boosting
6. **CatBoost Regressor**: Gradient boosting with categorical features
7. **AdaBoost Regressor**: Adaptive boosting ensemble

### Model Selection

- Models are evaluated using **R² score** (coefficient of determination)
- Hyperparameter tuning is performed using cross-validation
- The best performing model is automatically selected and saved

### Preprocessing

- **Numerical Features** (reading score, writing score):
  - Missing value imputation (median)
  - Standard scaling (mean=0, std=1)

- **Categorical Features** (gender, ethnicity, etc.):
  - Missing value imputation (most frequent)
  - One-hot encoding
  - Standard scaling (without centering)

## 📡 API Documentation

### Web Routes

#### `GET /`
- **Description**: Landing page
- **Response**: Renders `index.html`

#### `GET /predictdata`
- **Description**: Prediction form page
- **Response**: Renders `home.html` with empty form

#### `POST /predictdata`
- **Description**: Submit prediction request
- **Request Body** (form data):
  - `gender`: "male" or "female"
  - `ethnicity`: "group A", "group B", "group C", "group D", or "group E"
  - `parental_level_of_education`: "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", or "some high school"
  - `lunch`: "free/reduced" or "standard"
  - `test_preparation_course`: "none" or "completed"
  - `reading_score`: Number (0-100)
  - `writing_score`: Number (0-100)
- **Response**: Renders `home.html` with prediction result

### Python API

#### `CustomData` Class

```python
CustomData(
    gender: str,
    race_ethnicity: str,
    parental_level_of_education: str,
    lunch: str,
    test_preparation_course: str,
    reading_score: int,
    writing_score: int
)
```

**Methods**:
- `get_data_as_frame()`: Returns a pandas DataFrame with the input data

#### `PredictPipeline` Class

```python
PredictPipeline()
```

**Methods**:
- `predict(features: pd.DataFrame)`: Returns numpy array of predictions

## 🔧 Development

### Running Tests

Currently, the project uses logging for debugging. Check the `logs/` directory for detailed execution logs.

### Adding New Models

To add a new model to the training pipeline:

1. Import the model in `src/components/model_trainer.py`
2. Add it to the `models` dictionary
3. Add hyperparameters to the `params` dictionary
4. The system will automatically include it in evaluation

### Modifying Preprocessing

Edit `src/components/data_transformation.py` to modify:
- Feature selection
- Encoding strategies
- Scaling methods
- Missing value handling

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add comments for complex logic
- Maintain consistent indentation (4 spaces)

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Farouk**

- Email: abdelrahman.farouk56@gmail.com
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- Dataset: Student Performance Dataset
- Libraries: scikit-learn, Flask, pandas, and the open-source ML community

## 📊 Performance Metrics

The model performance is evaluated using:
- **R² Score**: Measures the proportion of variance explained by the model
- **Cross-Validation**: Ensures model generalization

Check the training logs in `logs/` directory for detailed performance metrics of each model.

## 🔍 Troubleshooting

### Common Issues

1. **Model file not found**:
   - Ensure you've run the training pipeline first
   - Check that `artifacts/model.pkl` exists

2. **Data file not found**:
   - Verify `notebook/data/data.csv` exists
   - Check file path in `data_ingestion.py`

3. **Import errors**:
   - Activate your virtual environment
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Port already in use**:
   - Change the port in `app.py`: `app.run(host='0.0.0.0', port=5001)`

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Note**: This is an educational project. For production use, consider adding:
- Input validation and sanitization
- Authentication and authorization
- API rate limiting
- Comprehensive testing suite
- CI/CD pipeline
- Docker containerization
- Database integration for storing predictions
