# Real Estate Price Prediction using Random Forest Regression

## Overview
This project implements a real estate price prediction model using the Random Forest Regression algorithm. The model is trained on a dataset containing various features of real estate properties such as location, size, number of rooms, and other relevant attributes. The goal is to predict the price of a property based on these features.

## Features
- Data preprocessing (handling missing values, encoding categorical features, feature scaling)
- Exploratory Data Analysis (EDA) with visualizations
- Training and evaluating a Random Forest Regression model
- Hyperparameter tuning for better accuracy
- Predicting real estate prices

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset
The dataset used contains the following attributes:
- `Location` - The area where the property is located
- `Size` - The total size of the property (e.g., in square feet)
- `Bedrooms` - Number of bedrooms
- `Bathrooms` - Number of bathrooms
- `Floors` - Number of floors
- `Age` - Age of the property
- `Price` - The target variable (property price)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/real-estate-price-prediction.git
   cd real-estate-price-prediction
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Load and preprocess the dataset.
3. Train the Random Forest Regression model.
4. Evaluate the model and visualize results.
5. Use the trained model to predict property prices.

## Model Training
- The dataset is split into training and testing sets.
- The Random Forest Regression model is trained on the training data.
- Hyperparameters like the number of trees (`n_estimators`), depth (`max_depth`), and others are tuned to optimize performance.
- The model is evaluated using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Results
The trained model achieves high accuracy in predicting property prices based on historical data. Feature importance analysis is used to identify key factors influencing price predictions.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.
git 
## Contact
For any inquiries, contact: [matyostsegay@gmail.com](mailto:matyostsegay@gmail.com)
