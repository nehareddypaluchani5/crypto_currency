import requests

def check_crypto_api():
    # Define potential APIs for testing
    api_urls = [
        'https://api.coingecko.com/api/v3/coins/markets',
        'https://api.coinpaprika.com/v1/tickers',
        'https://api.binance.com/api/v3/ticker/24hr'
    ]
    
    api_details = {}
    
    for url in api_urls:
        try:
            # Testing if the API is accessible
            response = requests.get(url, params={'vs_currency': 'usd'})
            response.raise_for_status()
            data = response.json()
            
            # Check number of available crypto pairs
            pairs = set([coin['symbol'] for coin in data])
            api_details[url] = {
                "Number of pairs supported": len(pairs),
                "Sample pairs": list(pairs)[:5],  # Displaying a few for sample
            }

            # Verify if daily data is available
            if 'timestamp' in data[0]:
                api_details[url]["Timeframes available"] = "Daily data available"
            else:
                api_details[url]["Timeframes available"] = "Unclear or Not Available"
                
            # Checking date availability range
            # This can be specific per API, adjusting to fetch further history if available
            # Here we assume Binance-like endpoint supports some range queries.
            if 'date_available' in response.headers:
                api_details[url]["Date availability range"] = response.headers['date_available']
            else:
                api_details[url]["Date availability range"] = "Range not specified in this API"
                
        except Exception as e:
            print(f"Failed to fetch data from {url}: {e}")
            api_details[url] = "Not accessible or data format incompatible"
    
    return api_details

# Execute and print the results
crypto_api_info = check_crypto_api()
print("API Information Summary:")
for api, details in crypto_api_info.items():
    print(f"\nAPI: {api}")
    for detail, info in details.items():
        print(f"  {detail}: {info}")





import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_data(crypto_pair, start_date):
    """
    Fetches daily historical data for a specified cryptocurrency pair starting from a given date.
    
    Args:
        crypto_pair (str): The cryptocurrency pair (e.g., "BTC/USD").
        start_date (str): Start date in 'YYYY-MM-DD' format.
    
    Returns:
        DataFrame: A DataFrame containing Date, Open, High, Low, and Close prices.
    """
    # Convert start_date to a datetime object
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.now()

    # Split the crypto_pair (e.g., BTC/USD -> btc and usd)
    crypto, currency = crypto_pair.split('/')

    # List to collect daily data
    all_data = []

    # API base URL for historical data from CoinGecko
    api_url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart/range"
    
    # Loop through each day (CoinGecko allows max 90 days range per request)
    while start_date_dt < end_date_dt:
        # Define the end date for this chunk (up to 90 days from start date)
        chunk_end_date_dt = min(start_date_dt + timedelta(days=90), end_date_dt)
        
        # Convert dates to UNIX timestamps for API query
        start_timestamp = int(start_date_dt.timestamp())
        end_timestamp = int(chunk_end_date_dt.timestamp())
        
        try:
            # Fetch data for this date range
            response = requests.get(api_url, params={
                "vs_currency": currency.lower(),
                "from": start_timestamp,
                "to": end_timestamp
            })
            response.raise_for_status()
            
            # Extract the data
            prices = response.json().get("prices", [])
            
            # Process each data point
            for price_data in prices:
                date = datetime.fromtimestamp(price_data[0] / 1000).strftime('%Y-%m-%d')
                open_price = price_data[1]
                
                # Example to fetch high, low, close from API (if provided by API endpoint)
                # For simplicity here, assume high, low, close are similar to open price
                # Replace with real values if your API provides it
                
                all_data.append({
                    "Date": date,
                    "Open": open_price,
                    "High": open_price,  # Replace with actual high value if available
                    "Low": open_price,   # Replace with actual low value if available
                    "Close": open_price  # Replace with actual close value if available
                })
        
        except requests.RequestException as e:
            print(f"Error fetching data for {crypto_pair}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error
        
        # Move start_date to the next day after the chunk end date
        start_date_dt = chunk_end_date_dt + timedelta(days=1)

    # Convert the list to a DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure the data is sorted by date
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df

# Example usage
crypto_pair = "bitcoin/usd"  # Use 'bitcoin' as per CoinGecko's naming convention for BTC
start_date = "2023-01-01"
df = fetch_crypto_data(crypto_pair, start_date)
print(df.head())



import pandas as pd
import numpy as np

def calculate_metrics(data, variable1, variable2):
    """
    Calculates historical and future high/low metrics for a cryptocurrency dataset.
    
    Args:
        data (DataFrame): DataFrame containing historical crypto data with columns 'Date', 'Open', 'High', 'Low', 'Close'.
        variable1 (int): Look-back period in days for calculating historical metrics.
        variable2 (int): Look-forward period in days for calculating future metrics.
    
    Returns:
        DataFrame: Original DataFrame with additional metric columns.
    """
    
    # Ensure the data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)

    # Calculate Historical High, Days Since High, and % Difference from Historical High
    data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1, min_periods=1).max()
    data[f'Days_Since_High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1, min_periods=1).apply(
        lambda x: (len(x) - 1) - np.argmax(x), raw=True)
    data[f'%_Diff_From_High_Last_{variable1}_Days'] = ((data['Close'] - data[f'High_Last_{variable1}_Days']) / data[f'High_Last_{variable1}_Days']) * 100

    # Calculate Historical Low, Days Since Low, and % Difference from Historical Low
    data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1, min_periods=1).min()
    data[f'Days_Since_Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1, min_periods=1).apply(
        lambda x: (len(x) - 1) - np.argmin(x), raw=True)
    data[f'%_Diff_From_Low_Last_{variable1}_Days'] = ((data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[f'Low_Last_{variable1}_Days']) * 100

    # Calculate Future High and % Difference from Future High
    data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2, min_periods=1).max()
    data[f'%_Diff_From_High_Next_{variable2}_Days'] = ((data['Close'] - data[f'High_Next_{variable2}_Days']) / data[f'High_Next_{variable2}_Days']) * 100

    # Calculate Future Low and % Difference from Future Low
    data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2, min_periods=1).min()
    data[f'%_Diff_From_Low_Next_{variable2}_Days'] = ((data['Close'] - data[f'Low_Next_{variable2}_Days']) / data[f'Low_Next_{variable2}_Days']) * 100

    # Handle any potential NaN values
    data.fillna(0, inplace=True)

    return data

# Example usage
# Assuming 'df' is the DataFrame returned from fetch_crypto_data function
variable1 = 7  # Look-back period in days
variable2 = 5  # Look-forward period in days
df_with_metrics = calculate_metrics(df, variable1, variable2)
print(df_with_metrics.head())




# ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class CryptoPredictor:
    def __init__(self, data):
        """
        Initialize the model with historical crypto data.
        Args:
            data (DataFrame): Data containing the required features and target variables.
        """
        # Define features and target variables for model training
        self.features = [
            'Days_Since_High_Last_variable1_Days',
            '%_Diff_From_High_Last_variable1_Days',
            'Days_Since_Low_Last_variable1_Days',
            '%_Diff_From_Low_Last_variable1_Days'
        ]
        self.target_high = '%_Diff_From_High_Next_variable2_Days'
        self.target_low = '%_Diff_From_Low_Next_variable2_Days'
        
        # Clean and prepare the data
        self.data = data.dropna(subset=self.features + [self.target_high, self.target_low])
        
        # Initialize the model
        self.model_high = LinearRegression()
        self.model_low = LinearRegression()

    def train_model(self):
        """
        Train the model to predict future high and low price differences.
        Returns:
            dict: Accuracy scores for both high and low price prediction models.
        """
        # Prepare training and test sets
        X = self.data[self.features]
        y_high = self.data[self.target_high]
        y_low = self.data[self.target_low]
        
        X_train, X_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
            X, y_high, test_size=0.2, random_state=42
        ), train_test_split(
            X, y_low, test_size=0.2, random_state=42
        )
        
        # Train models
        self.model_high.fit(X_train, y_high_train)
        self.model_low.fit(X_train, y_low_train)

        # Evaluate models
        high_preds = self.model_high.predict(X_test)
        low_preds = self.model_low.predict(X_test)
        
        high_accuracy = r2_score(y_high_test, high_preds)
        low_accuracy = r2_score(y_low_test, low_preds)
        
        return {
            "High Prediction R2 Score": high_accuracy,
            "Low Prediction R2 Score": low_accuracy
        }

    def predict_outcomes(self, new_data):
        """
        Predict the future high and low price differences.
        Args:
            new_data (dict): Dictionary with feature values.
        
        Returns:
            dict: Predicted % difference for high and low prices.
        """
        X_new = pd.DataFrame([new_data])
        high_pred = self.model_high.predict(X_new[self.features])[0]
        low_pred = self.model_low.predict(X_new[self.features])[0]
        
        return {
            "Predicted % Diff From High": high_pred,
            "Predicted % Diff From Low": low_pred
        }

# Example usage:
# Assuming 'df_with_metrics' is the DataFrame with metrics calculated in Step 3
predictor = CryptoPredictor(df_with_metrics)
accuracy_scores = predictor.train_model()
print("Model Accuracy Scores:", accuracy_scores)

new_data = {
    "Days_Since_High_Last_variable1_Days": 3,
    "%_Diff_From_High_Last_variable1_Days": -1.5,
    "Days_Since_Low_Last_variable1_Days": 4,
    "%_Diff_From_Low_Last_variable1_Days": 2.0
}
predictions = predictor.predict_outcomes(new_data)
print("Predictions:", predictions)


# Export to Excel
df_with_metrics.to_excel("crypto_data_metrics.xlsx", index=False, sheet_name="Crypto Metrics")







