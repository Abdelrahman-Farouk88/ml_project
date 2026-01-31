# API Documentation

## Overview

This document provides detailed API documentation for the Student Exam Performance Predictor application.

## Web API Endpoints

### 1. Landing Page

**Endpoint**: `GET /`

**Description**: Returns the landing page of the application.

**Response**: HTML page (`index.html`)

**Example**:
```bash
curl http://localhost:5000/
```

---

### 2. Prediction Form

**Endpoint**: `GET /predictdata`

**Description**: Returns the prediction form page.

**Response**: HTML page (`home.html`) with empty form

**Example**:
```bash
curl http://localhost:5000/predictdata
```

---

### 3. Submit Prediction

**Endpoint**: `POST /predictdata`

**Description**: Submits student data and returns predicted math score.

**Content-Type**: `application/x-www-form-urlencoded`

**Request Parameters**:

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `gender` | string | Yes | Student's gender | `male`, `female` |
| `ethnicity` | string | Yes | Race or ethnicity group | `group A`, `group B`, `group C`, `group D`, `group E` |
| `parental_level_of_education` | string | Yes | Parent's education level | `associate's degree`, `bachelor's degree`, `high school`, `master's degree`, `some college`, `some high school` |
| `lunch` | string | Yes | Lunch type | `free/reduced`, `standard` |
| `test_preparation_course` | string | Yes | Test preparation status | `none`, `completed` |
| `reading_score` | float | Yes | Reading score (0-100) | 0.0 to 100.0 |
| `writing_score` | float | Yes | Writing score (0-100) | 0.0 to 100.0 |

**Response**: HTML page (`home.html`) with prediction result

**Example Request**:
```bash
curl -X POST http://localhost:5000/predictdata \
  -d "gender=female" \
  -d "ethnicity=group B" \
  -d "parental_level_of_education=bachelor's degree" \
  -d "lunch=standard" \
  -d "test_preparation_course=none" \
  -d "reading_score=72" \
  -d "writing_score=74"
```

**Example Response** (HTML):
```html
<div class="result-container">
    <h2>Predicted Math Score</h2>
    <div class="result-value">73.45 / 100</div>
</div>
```

---

## Python API

### CustomData Class

**Location**: `src.pipeline.predict_pipeline.CustomData`

**Purpose**: Encapsulates input data for predictions.

**Constructor**:
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

**Parameters**:
- `gender` (str): Student's gender
- `race_ethnicity` (str): Race/ethnicity group
- `parental_level_of_education` (str): Parent's education level
- `lunch` (str): Lunch type
- `test_preparation_course` (str): Test preparation status
- `reading_score` (int): Reading score (0-100)
- `writing_score` (int): Writing score (0-100)

**Methods**:

#### `get_data_as_frame()`

Converts the input data to a pandas DataFrame.

**Returns**: `pandas.DataFrame`

**Example**:
```python
from src.pipeline.predict_pipeline import CustomData

data = CustomData(
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='none',
    reading_score=72,
    writing_score=74
)

df = data.get_data_as_frame()
print(df)
```

---

### PredictPipeline Class

**Location**: `src.pipeline.predict_pipeline.PredictPipeline`

**Purpose**: Handles model loading and prediction.

**Constructor**:
```python
PredictPipeline()
```

**Methods**:

#### `predict(features)`

Makes predictions using the trained model.

**Parameters**:
- `features` (pandas.DataFrame): Input features for prediction

**Returns**: `numpy.ndarray` - Array of predicted math scores

**Example**:
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Prepare data
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
pipeline = PredictPipeline()
prediction = pipeline.predict(data.get_data_as_frame())
print(f"Predicted Math Score: {prediction[0]:.2f}")
```

---

## Error Handling

### Common Errors

1. **Model Not Found Error**
   - **Cause**: Model file (`artifacts/model.pkl`) doesn't exist
   - **Solution**: Run the training pipeline first

2. **Preprocessor Not Found Error**
   - **Cause**: Preprocessor file (`artifacts/preprocessor.pkl`) doesn't exist
   - **Solution**: Run the training pipeline first

3. **Invalid Input Error**
   - **Cause**: Invalid or missing form parameters
   - **Solution**: Ensure all required fields are provided with valid values

4. **Value Error**
   - **Cause**: Invalid data types or out-of-range values
   - **Solution**: Validate input data before submission

---

## Response Format

### Success Response

When a prediction is successful, the response includes:
- HTML page with the predicted score displayed
- Score format: `XX.XX / 100` (rounded to 2 decimal places)

### Error Response

Errors are handled by the Flask application and may return:
- HTML error page
- HTTP status codes (400, 500, etc.)
- Error messages in logs (check `logs/` directory)

---

## Rate Limiting

Currently, there are no rate limits implemented. For production use, consider implementing:
- Request rate limiting
- API key authentication
- Request throttling

---

## Notes

- The application runs on `http://localhost:5000` by default
- All predictions are made using the model saved in `artifacts/model.pkl`
- Input validation is performed on the client side (HTML5) and server side
- Logs are stored in the `logs/` directory with timestamps

