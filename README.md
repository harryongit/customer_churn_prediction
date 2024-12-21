# Customer Churn Prediction Project

## Overview
The **Customer Churn Prediction Project** aims to predict customer churn based on historical data. This project utilizes machine learning techniques, primarily Random Forest, to forecast whether a customer will leave the company or not. It not only demonstrates ML engineering expertise but also focuses on deriving business insights that can be actionable for customer retention strategies.

![img](https://github.com/user-attachments/assets/17f2817c-ac87-4308-ab11-fcbee182fe98)

## Features
- **Data Preprocessing & Feature Engineering**: Cleaning and transforming raw data into a format suitable for model training.
- **Machine Learning Model**: Implementation of the Random Forest algorithm for churn prediction.
- **Model Evaluation**: Use of performance metrics such as accuracy, precision, recall, and AUC to evaluate the model's performance.
- **Visualization**: Graphical representation of important metrics and model results to provide clarity on business insights.
- **Business Insights**: Actionable insights generated from model outputs to guide strategic decisions around customer retention.
- **Modular Code Structure**: Well-organized and modular code base for easier understanding and maintainability.
- **Unit Tests**: Ensuring the correctness and robustness of the code through unit testing.

## Installation

To get started, clone the repository and install the dependencies by running:

```bash
pip install -r requirements.txt
```

Ensure that you have Python (version 3.7 or higher) and `pip` installed.

## Usage

After installing the dependencies, you can run the project using the following command:

```bash
python main.py
```

This will trigger the full pipeline, from data preprocessing to model training, evaluation, and visualization.

### Optional: Running Jupyter Notebooks
For exploratory data analysis, you can explore the provided Jupyter notebooks located in the `notebooks/` directory. These notebooks contain detailed steps for analyzing the data and feature engineering.

## Project Structure

The project is organized into the following directory structure:

```
Customer_Churn_Prediction/
│
├── data/                         # Directory containing the customer dataset
│   └── customer_data.csv          # Raw customer dataset
│
├── notebooks/                    # Jupyter notebooks for exploratory data analysis (EDA)
│   └── exploratory_analysis.ipynb # EDA and visualizations of the dataset
│
├── src/                          # Source code for data preprocessing, model, and evaluation
│   ├── __init__.py               # Initialize the source code package
│   ├── data_preprocessing.py     # Script for cleaning and preprocessing data
│   ├── evaluation.py             # Script for evaluating the model's performance
│   ├── feature_engineering.py    # Script for feature extraction and selection
│   └── model.py                 # Script for model implementation and training
│
├── tests/                        # Unit tests for key modules
│   └── test_preprocessing.py     # Unit tests for data preprocessing
│
├── main.py                       # Main entry point to execute the churn prediction pipeline
├── requirements.txt              # List of required Python dependencies
├── LICENSE                       # Project license file
└── README.md                     # This README file

```

### Key Files:
- `data/`: This directory contains the raw and processed customer data.
- `notebooks/`: Jupyter notebooks to aid in exploratory data analysis (EDA) and visualizations.
- `src/`: Contains the source code that handles data preprocessing, feature engineering, model implementation, and performance evaluation.
- `tests/`: Contains unit tests to ensure the correctness of each module in the project.
- `main.py`: The entry point for executing the churn prediction pipeline.
- `requirements.txt`: A list of Python dependencies necessary for running the project.

## Contributing

Contributions to this project are welcome! If you have an idea for an improvement, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add new feature or fix'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request for review.

Before submitting a major change, please open an issue to discuss the potential change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
