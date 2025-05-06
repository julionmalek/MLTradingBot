"""
Enhanced ML + Technical Strategy with Lumibot

Features:
- Clusters observations into different "regimes" based on technical indicators (RSI, SMA, MACD, ADX, Bollinger Bands)
- Uses FinBERT sentiment analysis to gauge bullish/bearish sentiment. (NOT YET)
"""

# --- 1. IMPORT MODULES ---
if True:
    import time
    import pandas as pd
    import numpy as np
    import talib as ta
    import joblib
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import BayesianRidge


    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import log_loss
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    from hmmlearn import hmm
    from arch import arch_model
    from lumibot.backtesting import YahooDataBacktesting


# --- 2. DEFINE TRADER BOT CLASS ---
class AdvancedMLTrader:
    """
    A multi-factor strategy combining:
    The strategy invests 25% of capital into SPY at init (buy & hold),
    then trades other symbols based on technical + sentiment conditions.
    """

    def __init__(self, start, end, data_source, name, broker, benchmark, parameters, model = None, debug = True):
        self.start = start
        self.end = end
        self.data_source = data_source
        self.name = name
        self.broker = broker
        self.benchmark = benchmark
        self.stock_symbols = parameters["symbols"]
        
        self.model = model if model else LinearRegression() # You can pass any sklearn-like model; default is simple LinearRegression
        self.trained_models = {}  # store models if you want to inspect later
        self.predictions = []
        self.dates = []
        self.last_model = None  # Keep track of the last model parameters
        self.learning_rate = 0.1  # (for q-learning, in returns prediction) Define a learning rate for model updates
        self.q_gamma = 0.95  # (for q-learning, in returns prediction) Adjust this based on experimentation

        # Portfolio section:
        self.cash = parameters["cash"]
        self.holdings = {}  # Current stock holdings
        self.portfolio_weights = {}  # Current target weights
        self.weight_history = {}  # Historical weights (optional)
        self.portfolio_returns = []  # Daily portfolio returns
        self.portfolio_values = []  # Daily total value of portfolio
        self.daily_predictions = {}  # {date: {symbol: predicted_return}}
        self.prediction_history = {}  # Keep full history
        self.sharpe_ratios = []  # Sharpe ratios over time
        self.risk_aversion = parameters.get("risk aversion", 1.0)

    def initialize_portfolio(self):
        """
        Sets up initial portfolio weights (1/n per symbol) and computes initial holdings
        based on the latest available prices in self.data_source.
        """
        symbols = self.stock_symbols
        n = len(symbols)
        equal_weight = 1 / n
        self.portfolio_weights = {symbol: equal_weight for symbol in symbols}

        for symbol in symbols:
            # Get latest available price
            try:
                price_data = self.data_source[symbol]
                if isinstance(price_data, pd.DataFrame):
                    latest_price = price_data.iloc[-1]["close"]
                else:
                    latest_price = price_data[-1].close  # fallback for list-like
            except Exception as e:
                print(f"Error retrieving price for {symbol}: {e}")
                continue

            # Allocate cash based on equal weights
            allocated_cash = self.cash * equal_weight
            quantity = allocated_cash // latest_price
            self.holdings[symbol] = {
                "quantity": quantity,
                "price": latest_price,
                "value": quantity * latest_price
            }

        if self.debug:
            print(f"Initialized portfolio weights: {self.portfolio_weights}")
            print(f"Initial holdings: {self.holdings}")

    def generate_features(self, regime_labels = None, compute_garch = False, garch_vols = None, compute_daily_sharpe=False, daily_sharpe=None, rolling_window = 90, return_combined = False, return_raw_targets = False):
        """
        Given raw OHLCV data, compute features for forecasting and clustering.

        Parameters:
            data (dict): Dictionary of {symbol: DataFrame}.
            regime_labels (Series, optional): Precomputed regime labels.
            garch_vols (Series, optional): Precomputed GARCH volatilities.
            rolling_window (int): Rolling window for averaging returns.
        """
        stock_dfs = []
        feature_types = ['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']

        for symbol, df_raw in self.data_source.items():
            print(f"{symbol}: {df_raw.df.shape}")

            df = df_raw.df.copy()
            df.columns = df.columns.str.lower()
            df = df[(df.index >= self.start) & (df.index <= self.end)].copy()

            df['symbol'] = symbol
            df['daily_return'] = df['close'].pct_change()
            df['volatility_5d'] = df['daily_return'].rolling(5).std()
            df['RSI'] = ta.RSI(df['close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod = 12, slowperiod = 26, signalperiod = 9)
            df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod = 14)
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['close'], timeperiod = 20, nbdevup = 2, nbdevdn = 2, matype = 0)
            df['boll_width'] = df['BB_upper'] - df['BB_lower']

            df = df[['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']]
            stock_dfs.append(df)

        # Combine across stocks
        combined = pd.concat(stock_dfs, axis=1, keys = self.data_source.keys()).dropna()
        combined.columns = [f"{symbol}_{feat}" for symbol, feat in combined.columns]

        # Average features across symbols
        averaged_features = pd.DataFrame(index=combined.index)
        for ft in feature_types:
            cols = [col for col in combined.columns if ft in col]
            averaged_features[ft] = combined[cols].mean(axis = 1)

        # Add rolling average of daily returns
        averaged_features['daily_return_rolling'] = averaged_features['daily_return'].rolling(window = rolling_window).mean()

        # Add regime if available
        if regime_labels is not None:
            averaged_features['regime'] = regime_labels

        # Add GARCH volatility if available
        if garch_vols is not None:
            averaged_features['garch_vol'] = garch_vols
        elif compute_garch:
            #averaged_features['garch_vol'] = self.forecast_garch_volatility(averaged_features['daily_return'])

            garch_vol = self.forecast_garch_volatility(averaged_features['daily_return'])
            print("GARCH volatility head:")
            print(garch_vol.head())
            print("Any NaNs in garch_vol?", garch_vol.isna().sum())
            print("Length of garch_vol:", len(garch_vol))
            print("Length of averaged_features:", len(averaged_features))

            # Align index just in case
            garch_vol = garch_vol.reindex(averaged_features.index)

            # Optional: don't drop rows just because GARCH is NaN
            averaged_features['garch_vol'] = garch_vol

         # Add daily Sharpe if available
        if daily_sharpe is not None:
            averaged_features['daily_sharpe'] = daily_sharpe
        elif compute_daily_sharpe:
            daily_sharpe_series = self.compute_sharpe_rolling()
            daily_sharpe_series = daily_sharpe_series.reindex(averaged_features.index)
            averaged_features['daily_sharpe'] = daily_sharpe_series

        raw_targets = averaged_features[['daily_return', 'garch_vol']] if 'garch_vol' in averaged_features.columns else None

        # Drop missing values
        averaged_features = averaged_features.dropna()
        print(f"Number of rows in averaged_features after dropna: {len(averaged_features)}")


        # Normalize (important for PCA, optional for time series regression)
        scaler = StandardScaler().fit(averaged_features)
        X_scaled = scaler.transform(averaged_features)
        averaged_features_scaled = pd.DataFrame(X_scaled, index=averaged_features.index, columns=averaged_features.columns)
        
        if return_combined and return_raw_targets:
            return averaged_features_scaled, scaler, combined, raw_targets
        
        elif return_combined and not return_raw_targets:
            return averaged_features_scaled, scaler, combined
        
        elif return_raw_targets and not return_combined:
            return averaged_features_scaled, scaler, raw_targets
        
        else:
            return averaged_features_scaled, scaler


    def optimal_cluster_params(self, n_components_slider_range=(2, 4, 1), n_regimes_slider_range=(2, 4, 1), rolling_window_slider_range=(30, 120, 30)):
        """
        Optimize the parameters for n_components, n_regimes, rolling_window, and weighting_options
        based on the Maximum Likelihood Estimate (MLE).
        
        Parameters:
            n_components_slider_range (tuple): The range (min, max, step) for n_components.
            n_regimes_slider_range (tuple): The range (min, max, step) for n_regimes.
            rolling_window_slider_range (tuple): The range (min, max, step) for rolling_window.

        Returns:
            dict: Optimal parameters {'n_components': [...], 'n_regimes': [...], 'rolling_window': [...], 'weighting_options': [...]}.
        """
        print("Running optimal_params...")  # <-- Add this to confirm the function is called
        time.sleep(4)
        # Create ranges based on slider inputs
        n_components_range = list(range(n_components_slider_range[0], n_components_slider_range[1] + 1, n_components_slider_range[2]))
        n_regimes_range = list(range(n_regimes_slider_range[0], n_regimes_slider_range[1] + 1, n_regimes_slider_range[2]))
        rolling_window_range = list(range(rolling_window_slider_range[0], rolling_window_slider_range[1] + 1, rolling_window_slider_range[2]))
        
        weighting_options_range = [
            {'time': True, 'volatility': False, 'portfolio': False},
            {'time': False, 'volatility': True, 'portfolio': False},
            {'time': False, 'volatility': False, 'portfolio': True},
            {'time': True, 'volatility': True, 'portfolio': False},
            {'time': True, 'volatility': False, 'portfolio': True},
            {'time': False, 'volatility': True, 'portfolio': True},
            {'time': True, 'volatility': True, 'portfolio': True},
        ]

        best_mle = -float('inf')
        best_params = {}

        # Loop through all combinations of the parameters
        for n_components in n_components_range:
            for n_regimes in n_regimes_range:
                for rolling_window in rolling_window_range:
                    for weighting_options in weighting_options_range:
                        print(f"Testing params: n_components={n_components}, n_regimes={n_regimes}, rolling_window={rolling_window}, weighting_options={weighting_options}")
                        
                        try:
                            # Run cluster_analysis with the current parameter combination
                            averaged_features, eig_vecs, eig_vals, X_pca = self.cluster_analysis(
                                n_components=n_components,
                                n_regimes=n_regimes,
                                rolling_window = rolling_window,
                                return_combined = False,
                                weighting_options=weighting_options
                            )
                            
                            # Calculate the log-likelihood of the model (this is a basic example using the GaussianMixture)
                            model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
                            model.fit(X_pca)
                            log_likelihood = model.score(X_pca)  # This is the MLE for the HMM model

                            # If this is the best log-likelihood, store the parameters
                            if log_likelihood > best_mle:
                                best_mle = log_likelihood
                                best_params = {
                                    'n_components': n_components,
                                    'n_regimes': n_regimes,
                                    'rolling_window': rolling_window,
                                    'weighting_options': weighting_options
                                }
                        except Exception as e:
                            print(f"Error occurred for params {n_components}, {n_regimes}, {rolling_window}, {weighting_options}: {e}")

        print(f'[BEST PARAMETERS FOR CLUSTERING:] {best_params}')

        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        return best_params

    def cluster_analysis(self, n_components=3, n_regimes=3, rolling_window = 90, weighting_options=None, regime_labels=None, garch_vols=None, return_combined=False, return_raw_targets = False):
        """
        Run PCA with optional weighting, HMM clustering, and return regime information.

        Parameters:
            n_components (int): Number of principal components for PCA.
            n_regimes (int): Number of HMM regimes (clusters).
            rolling_window (int): Size of rolling window for features.
            return_combined (bool): Whether to return the full combined feature matrix.
            weighting_options (dict): Dictionary of booleans for 'time', 'volatility', 'portfolio' weighting.

        Returns:
            averaged_features, combined, pca_model, X_pca (if return_combined is True)
        """

        if weighting_options is None:
            weighting_options = {
                "time": False,
                "volatility": False,
                "portfolio": False
            }

        if return_combined and return_raw_targets:        
            averaged_features, scaler, combined, raw_targets = self.generate_features(regime_labels = regime_labels, garch_vols = garch_vols, rolling_window = rolling_window, return_combined = return_combined, return_raw_targets = return_raw_targets)

        elif return_combined and not return_raw_targets:
            averaged_features, scaler, combined = self.generate_features(regime_labels = regime_labels, garch_vols = garch_vols, rolling_window = rolling_window, return_combined = return_combined)

        elif return_raw_targets and not return_combined:
            averaged_features, scaler, raw_targets = self.generate_features(regime_labels = regime_labels, garch_vols = garch_vols, rolling_window = rolling_window, return_combined = return_combined)

        else:
            averaged_features, scaler = self.generate_features(regime_labels = regime_labels, garch_vols = garch_vols, rolling_window = rolling_window, return_combined = return_combined)
        
        X_scaled = averaged_features.values

        # Construct weights
        weights = np.ones(X_scaled.shape[0])

        if weighting_options.get("time", False):
            time_weights = np.linspace(0.1, 1, num=len(weights))
            weights *= time_weights

        if weighting_options.get("volatility", False):
            vol = averaged_features['volatility_5d'].values
            vol_weights = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)
            weights *= vol_weights

        if weighting_options.get("portfolio", False):
            portfolio_weights = averaged_features['daily_return_rolling'].abs().values
            weights *= portfolio_weights

        # Normalize weights
        weights /= np.sum(weights)

        # Weighted covariance matrix and PCA
        mean_centered = X_scaled - np.average(X_scaled, axis=0, weights=weights)
        cov_matrix = np.cov(mean_centered.T, aweights=weights)
        #print("cov_matrix shape:", cov_matrix.shape)  # Should be (6, 6)

        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        #print("eig_vecs shape:", eig_vecs.shape)  # Should be (6, n_components), where n_components = 2
        

        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, idx[:n_components]]
        X_pca = mean_centered @ eig_vecs

        # HMM
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
        model.fit(X_pca)
        hidden_states = model.predict(X_pca)

        averaged_features['regime'] = hidden_states

        joblib.dump(scaler, 'regime_scaler.pkl')
        joblib.dump(model, 'regime_hmm.pkl')
        averaged_features.to_csv("portfolio_pca_hmm_regimes.csv")

        if return_combined:
            return averaged_features, eig_vecs, eig_vals, X_pca, combined
        else:
            return averaged_features, eig_vecs, eig_vals, X_pca

    def forecast_garch_volatility(self, returns, p=1, q=1, window_size=5):
        """
        Forecast one-step-ahead GARCH(p,q) volatility using a rolling window.

        Parameters:
            returns (pd.Series): Daily returns.
            p, q (int): GARCH model parameters.
            window_size (int): Rolling window size to fit GARCH.

        Returns:
            pd.Series: Forecasted volatilities aligned with returns index.
        """
        print("Fitting GARCH model with rolling window...")

        from arch import arch_model

        returns = returns.dropna() * 100  # Scale returns like before
        vol_forecast = pd.Series(index=returns.index, dtype=float)

        for i in range(window_size, len(returns)):
            window = returns.iloc[i - window_size:i]
            date = returns.index[i]

            try:
                model = arch_model(window, vol='GARCH', p=p, q=q, rescale=False)
                res = model.fit(disp='off', show_warning=False)
                forecast = res.forecast(horizon=1)
                variance = forecast.variance.values[-1, 0]
                vol_forecast.loc[date] = np.sqrt(variance) / 100  # Convert back from %
            except Exception as e:
                print(f"GARCH failed at index {i} ({date}): {e}")
                continue

        print(f"Generated {vol_forecast.count()} GARCH volatility forecasts.")
        return vol_forecast


# Predict and update (old)
    
    def predict_and_update(self, features, targets, min_train_size=10):
        """
        Expanding window prediction using Q-learning inspired reward updates
        with polynomial feature expansion and Bayesian Ridge regression.

        Parameters:
            features (DataFrame): Feature matrix.
            targets (DataFrame): DataFrame with columns ['returns', 'volatility'].
            min_train_size (int): Minimum number of observations required to start predicting.
        """

        poly = PolynomialFeatures(degree=2, include_bias=False)
        features_poly = poly.fit_transform(features)

        n_obs = len(features)

        for t in range(min_train_size, n_obs):
            X_train = features_poly[:t+1]  # include current test point
            y_train = targets.iloc[:t+1].copy()

            X_test = features_poly[[t]]
            y_true = targets.iloc[t]

            if np.isnan(X_test).any() or y_true.isnull().values.any():
                continue

            # Predict using model trained on t points (before t)
            model = MultiOutputRegressor(BayesianRidge())
            model.fit(features_poly[:t], targets.iloc[:t])
            y_pred = model.predict(X_test)[0]

            if np.isnan(y_pred).any():
                continue

            # Q-learning inspired update
            target_adjustment = y_true.values + self.q_gamma * (y_true.values - y_pred)
            target_adjustment = y_pred + self.learning_rate * (target_adjustment - y_pred)

            # Re-train on expanding window including adjusted target
            X_train[-1] = X_test  # replace last row
            y_train.iloc[-1] = target_adjustment

            model = MultiOutputRegressor(BayesianRidge())
            model.fit(X_train, y_train)

            # Store
            self.predictions.append(y_pred)
            self.dates.append(features.index[t])
            self.trained_models[features.index[t]] = model
            self.last_model = model

            # Predict
            y_pred = model.predict(X_test)[0]

            # Skip if prediction produces NaN
            if np.isnan(y_pred).any():
                print(f"Skipping timestep {t} due to NaNs in prediction.")
                continue

            # Evaluate prediction
            true_return = y_true.iloc[0]
            true_vol = y_true.iloc[1]
            mse_return = mean_squared_error([true_return], [y_pred[0]])
            mse_volatility = mean_squared_error([true_vol], [y_pred[1]])
            reward = - (mse_return + mse_volatility)

            # Q-learning style update (temporal difference): Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            # In this regression setting, we approximate this as a reward-weighted learning step
            target_adjustment = y_true.values + self.q_gamma * (y_true.values - y_pred)
            target_adjustment = y_pred + self.learning_rate * (target_adjustment - y_pred)  # Soft update toward adjusted value

            # Refit model from scratch (BayesianRidge does not support partial_fit)
            try:
                X_train_updated = np.vstack([X_train, X_test])
                y_train_updated = np.vstack([y_train, target_adjustment.reshape(1, -1)])
                model = MultiOutputRegressor(BayesianRidge())
                model.fit(X_train_updated, y_train_updated)
            except Exception as e:
                print(f"Model update failed at timestep {t}: {e}")
                continue

            # Store and track
            self.predictions.append(y_pred)
            self.dates.append(features.index[t])
            self.trained_models[features.index[t]] = model
            self.last_model = model

            print(f"Predicted return: {y_pred[0]}, Predicted volatility: {y_pred[1]}")
            print(f"True return: {true_return}, True volatility: {true_vol}")
            print(f"MSE return: {mse_return}, MSE volatility: {mse_volatility}, Reward: {reward}")

# Evaluate model (old)
    
    def evaluate_model(self, actual_returns, actual_volatility):
        """
        Evaluate model performance by calculating MSE for returns and volatility.
        """
        # Extract predicted values from self.predictions
        predicted_returns = [pred[0] for pred in self.predictions]  # Assuming returns is the first element
        predicted_volatility = [pred[1] for pred in self.predictions]  # Assuming volatility is the second element

        # Ensure the predictions and actuals are aligned in length
        n = len(actual_returns)
        if n != len(predicted_returns):
            print("Error: Number of actual and predicted values do not match.")
            return

        # Calculate Mean Squared Error for returns
        mse_returns = mean_squared_error(actual_returns[-n:], predicted_returns)
        mse_volatility = mean_squared_error(actual_volatility[-n:], predicted_volatility)

        # Optionally, calculate R-squared as well
        r2_returns = r2_score(actual_returns[-n:], predicted_returns)
        r2_volatility = r2_score(actual_volatility[-n:], predicted_volatility)

        print(f"Mean Squared Error for Returns: {mse_returns}")
        print(f"Mean Squared Error for Volatility: {mse_volatility}")
        print(f"R² for Returns: {r2_returns}")
        print(f"R² for Volatility: {r2_volatility}")



'''
    def compute_sharpe_snapshot(self, date, risk_free_rate=0.0):
        """
        Compute a single-day Sharpe ratio snapshot for a specific date.
        Used for immediate feedback in reinforcement learning or tracking.
        """
        if date not in self.portfolio_weights:
            return None

        weights = self.portfolio_weights[date]
        daily_returns = {
            symbol: self.data_source[symbol].df['close'].pct_change().get(date)
            for symbol in self.stock_symbols
        }

        if any(pd.isna(ret) for ret in daily_returns.values()):
            return None

        port_return = sum(weights[symbol] * daily_returns[symbol] for symbol in self.stock_symbols)
        port_std = np.std(list(daily_returns.values()))
        sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0

        self.portfolio_returns.append(port_return)
        self.sharpe_ratios.append(sharpe)

        return sharpe

    def compute_sharpe_rolling(self, window: int = 21):
        """
        Compute a rolling Sharpe ratio time series.
        Used for feature generation and evaluation.

        Parameters:
            window (int): Rolling window for volatility estimation.

        Returns:
            pd.Series: Rolling Sharpe ratio.
        """
        weighted_returns = []

        for date in self.price_data.index:
            if date not in self.portfolio_weights:
                weighted_returns.append(np.nan)
                continue

            weights = self.portfolio_weights[date]
            daily_returns = {
                symbol: self.data_source[symbol].df['close'].pct_change().get(date)
                for symbol in self.stock_symbols
            }

            if any(pd.isna(r) for r in daily_returns.values()):
                weighted_returns.append(np.nan)
                continue

            port_return = sum(weights[symbol] * daily_returns[symbol] for symbol in self.stock_symbols)
            weighted_returns.append(port_return)

        returns_series = pd.Series(weighted_returns, index=self.price_data.index)
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        sharpe_series = rolling_mean / rolling_std

        return sharpe_series

    def predict_returns(self, features, targets, min_train_size=10):
        """
        Expanding window prediction using Q-learning inspired reward updates
        with polynomial feature expansion and Bayesian Ridge regression.
        Sharpe ratio is used to scale return adjustment only (not volatility).

        Parameters:
            features (DataFrame): Feature matrix.
            targets (DataFrame): DataFrame with columns ['returns', 'volatility'].
            min_train_size (int): Minimum number of observations required to start predicting.
        """
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features_poly = poly.fit_transform(features)
        n_obs = len(features)

        for t in range(min_train_size, n_obs):
            X_train = features_poly[:t+1]
            y_train = targets.iloc[:t+1].copy()
            X_test = features_poly[[t]]
            y_true = targets.iloc[t]

            if np.isnan(X_test).any() or y_true.isnull().values.any():
                continue

            model = MultiOutputRegressor(BayesianRidge())
            model.fit(features_poly[:t], targets.iloc[:t])
            y_pred = model.predict(X_test)[0]

            if np.isnan(y_pred).any():
                continue

            # === Sharpe-based reward scaling for return only ===
            recent_returns = targets['returns'].iloc[max(0, t - 19):t + 1]
            sharpe_reward = self.compute_sharpe_snapshot(recent_returns)

            if not np.isfinite(sharpe_reward):
                sharpe_reward = 1.0  # fallback to neutral scaling

            sharpe_reward = np.clip(sharpe_reward, 0.0, 3.0)  # avoid extreme scaling

            # Q-learning update:
            delta_return = y_true[0] - y_pred[0]
            adjusted_return = y_pred[0] + self.learning_rate * (y_true[0] + self.q_gamma * delta_return * sharpe_reward - y_pred[0])

            # Keep volatility prediction unchanged
            adjusted_volatility = y_true[1]

            # Apply the adjustment
            y_train.iloc[-1] = [adjusted_return, adjusted_volatility]

            model = MultiOutputRegressor(BayesianRidge())
            model.fit(X_train, y_train)

            self.predictions.append(y_pred)
            self.dates.append(features.index[t])
            self.trained_models[features.index[t]] = model
            self.last_model = model

            self.daily_predictions[features.index[t]] = dict(zip(self.stock_symbols, [adjusted_return] * len(self.stock_symbols)))

    def evaluate_performance(self, actual_returns, actual_volatility):
        predicted_returns = [pred[0] for pred in self.predictions]
        predicted_volatility = [pred[1] for pred in self.predictions]

        n = len(actual_returns)
        if n != len(predicted_returns):
            print("Mismatch between actual and predicted data lengths.")
            return

        mse_ret = mean_squared_error(actual_returns[-n:], predicted_returns)
        mse_vol = mean_squared_error(actual_volatility[-n:], predicted_volatility)
        r2_ret = r2_score(actual_returns[-n:], predicted_returns)
        r2_vol = r2_score(actual_volatility[-n:], predicted_volatility)

        # Rolling Sharpe ratio (you must have compute_sharpe_rolling already implemented)
        actual_series = pd.Series(actual_returns[-n:])
        sharpe_series = self.compute_sharpe_rolling(actual_series, window=20)
        avg_sharpe = sharpe_series.mean()

        print(f"MSE Returns: {mse_ret:.4f}, MSE Volatility: {mse_vol:.4f}")
        print(f"R² Returns: {r2_ret:.4f}, R² Volatility: {r2_vol:.4f}")
        print(f"Avg Rolling Sharpe (20-day): {avg_sharpe:.4f}")

        return mse_ret, mse_vol, r2_ret, r2_vol, avg_sharpe

    def adjust_portfolio_weights(self, current_date, risk_aversion=1.0):
        if current_date not in self.daily_predictions:
            print(f"No predictions available for {current_date}")
            return

        preds = self.daily_predictions[current_date]
        pred_returns = np.array([preds[symbol] for symbol in self.stock_symbols])
        returns_matrix = self.data_source[self.stock_symbols].loc[:current_date].pct_change().dropna()

        if returns_matrix.shape[0] < 2:
            print("Not enough data for covariance estimation.")
            return

        cov_matrix = returns_matrix.cov().values

        try:
            inv_cov = np.linalg.pinv(cov_matrix)
            raw_weights = inv_cov @ pred_returns
            norm_weights = raw_weights / np.sum(np.abs(raw_weights))
            adjusted_weights = norm_weights / risk_aversion
        except Exception as e:
            print(f"Weight optimization failed: {e}")
            adjusted_weights = np.ones(len(self.stock_symbols)) / len(self.stock_symbols)

        self.portfolio_weights[current_date] = dict(zip(self.stock_symbols, adjusted_weights))

'''


# Train and predict (old)
'''
    def train_and_predict_expanding(self, features, targets, min_train_size=100):
        """
        Expanding window training for multi-output regression.

        Parameters:
            features (DataFrame): Feature matrix.
            targets (DataFrame): DataFrame with columns ['returns', 'volatility'].
            min_train_size (int): Number of observations required before predictions start.
        """
        n_obs = len(features)

        # Initialize error tracking and performance evaluation storage
        self.errors = []  # To store the errors for each timestep
        self.evaluation_metrics = []  # To store performance metrics (e.g., MSE)

        for t in range(min_train_size, n_obs - 1):
            X_train = features.iloc[:t]
            y_train = targets.iloc[:t]
            X_test = features.iloc[[t]]
            y_true = targets.iloc[t]

            # Train multi-output regression model
            model = MultiOutputRegressor(LinearRegression())
            model.fit(X_train, y_train)

            # Predict both return and volatility
            y_pred = model.predict(X_test)[0]  # [pred_return, pred_volatility]

            # Save prediction and model
            self.predictions.append(y_pred)
            self.dates.append(features.index[t])
            self.trained_models[features.index[t]] = model

            # Calculate error (e.g., Mean Squared Error for both return and volatility)
            mse_return = mean_squared_error([y_true.iloc[0]], [y_pred[0]])  # Use iloc for positional indexing
            mse_volatility = mean_squared_error([y_true.iloc[1]], [y_pred[1]])  # Use iloc for positional indexing

            # Save the errors for future analysis
            self.errors.append({
                'date': features.index[t],
                'pred_return': y_pred[0],
                'true_return': y_true.iloc[0],  # Use iloc for positional indexing
                'return_mse': mse_return,
                'pred_volatility': y_pred[1],
                'true_volatility': y_true.iloc[1],  # Use iloc for positional indexing
                'volatility_mse': mse_volatility
            })

            # Calculate performance metrics (optional: you can add more metrics like R-squared)
            metrics = {
                'date': features.index[t],
                'mse_return': mse_return,
                'mse_volatility': mse_volatility,
                # You can include other metrics like R-squared or Sharpe ratio
            }

            self.evaluation_metrics.append(metrics)

            # Optionally, print or log the progress (this can be adjusted as per your needs)
            if t % 100 == 0:  # Print progress every 100 iterations (optional)
                print(f"Processed {t}/{n_obs} steps")

        print(f"Generated {len(self.errors)} predictions with evaluation metrics.")
'''

