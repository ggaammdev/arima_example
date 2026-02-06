# ARIMA Example

This project demonstrates how to use `pmdarima` (Auto-ARIMA) to forecast time series data in Python. It generates synthetic training data with a linear trend and seasonality to showcase the model's capabilities.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd arima_example
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to generate the synthetic data, train the model, and save the forecast plot:

```bash
python pdarima_ex.py
```

## Output

The script saves a visualization of the forecast to `arima_forecast.png`.

![ARIMA Forecast](./arima_forecast.png)
