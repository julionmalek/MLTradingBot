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
    from sklearn.linear_model import LinearRegression
    from hmmlearn import hmm
    from lumibot.backtesting import YahooDataBacktesting
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import log_loss
    #from arch import arch_model


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

    def generate_features(self, regime_labels = None, garch_vols = None, rolling_window = 90, return_combined = False):
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

        # Drop missing values
        averaged_features = averaged_features.dropna()

        # Normalize (important for PCA, optional for time series regression)
        scaler = StandardScaler().fit(averaged_features)
        X_scaled = scaler.transform(averaged_features)
        averaged_features_scaled = pd.DataFrame(X_scaled, index=averaged_features.index, columns=averaged_features.columns)
        
        if return_combined:
            return averaged_features_scaled, scaler, combined
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
                                rolling_window = 90,
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

    def cluster_analysis(self, n_components=3, n_regimes=3, rolling_window = 90, return_combined=False, weighting_options=None, regime_labels=None, garch_vols=None):
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
        '''
        stock_dfs = []

        # Later, it will be easier to have one method which calculates a list of macro indicators given a dataset.
        # Then, we can plug these into methods that run a type of analysis, e.g. clusters, GARCH...
        for key, df_raw in self.data_source.items():
            df = df_raw.df
            df['symbol'] = str(key)
            df.columns = df.columns.str.lower()
            df = df[(df.index >= self.start) & (df.index <= self.end)].copy()

            df['daily_return'] = df['close'].pct_change()
            df['volatility_5d'] = df['daily_return'].rolling(5).std()
            df['RSI'] = ta.RSI(df['close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['boll_width'] = df['BB_upper'] - df['BB_lower']

            df = df[['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']]
            stock_dfs.append(df)

        combined = pd.concat(stock_dfs, axis=1, keys=self.stock_symbols).dropna()
        combined.columns = [f"{symbol}_{feat}" for symbol, feat in combined.columns]

        averaged_features = pd.DataFrame(index=combined.index)
        feature_types = ['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']
        for ft in feature_types:
            cols = [col for col in combined.columns if ft in col]
            averaged_features[ft] = combined[cols].mean(axis=1)

        averaged_features['daily_return_rolling'] = averaged_features['daily_return'].rolling(window=rolling_window).mean()
        averaged_features.dropna(inplace=True)

        # Normalize features
        scaler = StandardScaler().fit(averaged_features)
        X_scaled = scaler.transform(averaged_features)
        #print("X_scaled shape:", X_scaled.shape)  # Should be (n_samples, 6)
        '''
        if return_combined:
            averaged_features, scaler, combined = self.generate_features(regime_labels = regime_labels, garch_vols = garch_vols, rolling_window = rolling_window, return_combined = return_combined)
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


