import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


@dataclass
class MarketStructureResult:
    as_of_date: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_assets: int
    n_observations: int
    tickers: List[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    demeaned_returns: pd.DataFrame
    covariance_matrix: pd.DataFrame
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    rmt_result: Dict[str, Any]
    participation_ratio_dimension: float
    levina_bickel_dimension: float


class SP500MarketStructure:
    """
    Research-grade class for historical S&P 500 market structure analysis.

    Features
    --------
    1. Load historical S&P 500 constituents from CSV
    2. Get constituents as of a given date
    3. Download price data
    4. Compute return, covariance, correlation matrices
    5. Perform eigendecomposition
    6. Estimate intrinsic dimensionality via:
        - Random Matrix Theory (Marchenko-Pastur threshold)
        - Participation Ratio
        - Levina-Bickel estimator
    7. Optional RMT covariance cleaning
    8. Plot spectrum and explained variance
    9. Rolling-window market dimension analysis
    """

    def __init__(
        self,
        constituents_csv_path: str,
        date_col: str = "date",
        tickers_col: str = "tickers"
    ):
        self.constituents_csv_path = constituents_csv_path
        self.date_col = date_col
        self.tickers_col = tickers_col

        self.constituents_df = self._load_constituents_data()

    # ------------------------------------------------------------------
    # Loading / preprocessing
    # ------------------------------------------------------------------
    def _load_constituents_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.constituents_csv_path)

        if self.date_col not in df.columns:
            raise ValueError(f"Missing date column '{self.date_col}' in CSV.")
        if self.tickers_col not in df.columns:
            raise ValueError(f"Missing tickers column '{self.tickers_col}' in CSV.")

        df = df[[self.date_col, self.tickers_col]].copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.normalize()

        def parse_tickers(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            return [t.strip() for t in str(x).split(",") if t.strip()]

        df[self.tickers_col] = df[self.tickers_col].apply(parse_tickers)
        df = df.sort_values(self.date_col).drop_duplicates(subset=[self.date_col]).reset_index(drop=True)

        return df

    def historical_constituents(self, as_of_date: str) -> List[str]:
        """
        Return S&P 500 constituents as of the most recent available date <= as_of_date.
        """
        target_date = pd.to_datetime(as_of_date).normalize()

        if target_date < self.constituents_df[self.date_col].min():
            raise ValueError("Requested date is earlier than the available constituent history.")

        eligible = self.constituents_df[self.constituents_df[self.date_col] <= target_date]
        if eligible.empty:
            raise ValueError("No constituent snapshot found on or before requested date.")

        row = eligible.iloc[-1]
        return row[self.tickers_col]

    # ------------------------------------------------------------------
    # Price / returns pipeline
    # ------------------------------------------------------------------
    def _normalize_yfinance_ticker(self, ticker: str) -> str:
        """
        yfinance generally prefers '-' instead of '.' for certain tickers.
        Example: BRK.B -> BRK-B
        """
        return ticker.replace(".", "-")

    def _download_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        yf_tickers = [self._normalize_yfinance_ticker(t) for t in tickers]

        data = yf.download(
            yf_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
            threads=True
        )

        if data.empty:
            raise ValueError("No price data returned from yfinance.")

        # MultiIndex case
        if isinstance(data.columns, pd.MultiIndex):
            level_values = list(data.columns.get_level_values(1).unique())

            if "Close" in level_values:
                prices = data.xs("Close", axis=1, level=1)
            elif "Adj Close" in level_values:
                prices = data.xs("Adj Close", axis=1, level=1)
            else:
                raise ValueError("Neither 'Close' nor 'Adj Close' found in downloaded data.")
        else:
            # Single ticker fallback
            if "Close" in data.columns:
                prices = data[["Close"]].copy()
                prices.columns = [yf_tickers[0]]
            elif "Adj Close" in data.columns:
                prices = data[["Adj Close"]].copy()
                prices.columns = [yf_tickers[0]]
            else:
                raise ValueError("Could not locate price column in downloaded data.")

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        return prices

    def get_price_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        min_nonmissing_fraction: float = 0.80,
        forward_fill: bool = True
    ) -> pd.DataFrame:
        """
        Download and clean prices.
        """
        prices = self._download_prices(tickers, start_date, end_date)

        if prices.shape[1] == 0:
            raise ValueError("No assets found in downloaded prices.")

        min_count = int(np.ceil(min_nonmissing_fraction * len(prices)))
        prices = prices.dropna(axis=1, thresh=min_count)

        if forward_fill:
            prices = prices.ffill()

        prices = prices.dropna(axis=0, how="any")

        if prices.shape[1] < 2:
            raise ValueError("Fewer than 2 valid assets remain after cleaning.")

        return prices

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        method: 'log' or 'simple'
        """
        if method not in {"log", "simple"}:
            raise ValueError("method must be 'log' or 'simple'.")

        if method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()

        return returns

    def build_matrices(
        self,
        returns: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        demeaned_returns = returns - returns.mean(axis=0)
        covariance_matrix = demeaned_returns.cov()
        correlation_matrix = returns.corr()
        return demeaned_returns, covariance_matrix, correlation_matrix

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------
    def eigendecomposition(
        self,
        matrix: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        evals, evecs = la.eigh(matrix.values)

        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        total = np.sum(evals)
        explained = evals / total if total > 0 else np.zeros_like(evals)
        cumulative = np.cumsum(explained)

        return evals, evecs, explained, cumulative

    # ------------------------------------------------------------------
    # Intrinsic dimension estimators
    # ------------------------------------------------------------------
    def rmt_dimension(
        self,
        eigenvalues: np.ndarray,
        demeaned_returns: pd.DataFrame,
        noise_estimation: str = "median_tail"
    ) -> Dict[str, Any]:
        """
        Estimate signal dimension using Marchenko-Pastur upper edge.

        q = p / n
        lambda_plus = sigma^2 * (1 + sqrt(q))^2

        Any empirical eigenvalue > lambda_plus is treated as signal.
        """
        X = demeaned_returns.values
        n, p = X.shape

        if n <= 1 or p <= 1:
            raise ValueError("Need at least 2 observations and 2 assets.")

        evals = np.sort(np.asarray(eigenvalues))[::-1]
        q = p / n

        if noise_estimation == "median_tail":
            tail = evals[max(1, p // 2):]
            sigma_squared = np.median(tail) if len(tail) > 0 else np.median(evals)
        elif noise_estimation == "mean_tail":
            tail = evals[max(1, p // 2):]
            sigma_squared = np.mean(tail) if len(tail) > 0 else np.mean(evals)
        else:
            raise ValueError("noise_estimation must be 'median_tail' or 'mean_tail'.")

        mp_upper = sigma_squared * (1 + np.sqrt(q)) ** 2
        mp_lower = sigma_squared * (1 - np.sqrt(q)) ** 2 if q <= 1 else 0.0

        signal_mask = evals > mp_upper
        signal_dimension = int(np.sum(signal_mask))

        return {
            "dimension": signal_dimension,
            "sigma_squared": float(sigma_squared),
            "q": float(q),
            "mp_lower_bound": float(mp_lower),
            "mp_upper_bound": float(mp_upper),
            "signal_eigenvalues": evals[signal_mask],
            "noise_eigenvalues": evals[~signal_mask]
        }

    def participation_ratio(
        self,
        eigenvalues: np.ndarray
    ) -> float:
        evals = np.asarray(eigenvalues)
        numerator = np.sum(evals) ** 2
        denominator = np.sum(evals ** 2)

        if denominator <= 0:
            raise ValueError("Invalid eigenvalue spectrum for participation ratio.")

        return float(numerator / denominator)

    def levina_bickel_dimension(
        self,
        demeaned_returns: pd.DataFrame,
        k: int = 10
    ) -> float:
        """
        Levina-Bickel intrinsic dimension estimator.
        Rows = time observations
        Columns = assets/features
        """
        X = np.asarray(demeaned_returns)
        n = X.shape[0]

        if k < 2:
            raise ValueError("k must be at least 2.")
        if k >= n:
            raise ValueError("k must be strictly less than the number of observations.")

        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)

        # distances[:,0] is self-distance = 0
        T_k = distances[:, k]
        T_j = distances[:, 1:k]

        eps = 1e-12
        logs = np.log((T_k[:, None] + eps) / (T_j + eps))
        local_dims = (k - 1) / np.sum(logs, axis=1)

        local_dims = local_dims[np.isfinite(local_dims)]
        if len(local_dims) == 0:
            raise ValueError("Levina-Bickel estimator failed: no finite local dimensions.")

        return float(np.mean(local_dims))

    # ------------------------------------------------------------------
    # Covariance cleaning
    # ------------------------------------------------------------------
    def rmt_clean_covariance(
        self,
        covariance_matrix: pd.DataFrame,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        rmt_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Replace all noise eigenvalues with their mean, preserve signal eigenvalues.
        """
        evals = np.asarray(eigenvalues).copy()
        mp_upper = rmt_result["mp_upper_bound"]

        signal_mask = evals > mp_upper
        noise_mask = ~signal_mask

        if np.sum(noise_mask) > 0:
            noise_mean = np.mean(evals[noise_mask])
            evals[noise_mask] = noise_mean

        cleaned = eigenvectors @ np.diag(evals) @ eigenvectors.T
        cleaned = (cleaned + cleaned.T) / 2.0

        return pd.DataFrame(
            cleaned,
            index=covariance_matrix.index,
            columns=covariance_matrix.columns
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(
        self,
        as_of_date: str,
        start_date: str,
        end_date: str,
        return_method: str = "log",
        min_nonmissing_fraction: float = 0.80,
        k_neighbors: int = 10,
        max_assets: Optional[int] = None
    ) -> MarketStructureResult:
        tickers = self.historical_constituents(as_of_date)

        if max_assets is not None:
            tickers = tickers[:max_assets]

        prices = self.get_price_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            min_nonmissing_fraction=min_nonmissing_fraction
        )

        returns = self.compute_returns(prices, method=return_method)
        demeaned_returns, covariance_matrix, correlation_matrix = self.build_matrices(returns)

        eigenvalues, eigenvectors, explained, cumulative = self.eigendecomposition(covariance_matrix)

        rmt_result = self.rmt_dimension(eigenvalues, demeaned_returns)
        pr_dim = self.participation_ratio(eigenvalues)

        try:
            lb_dim = self.levina_bickel_dimension(demeaned_returns, k=k_neighbors)
        except Exception as e:
            warnings.warn(f"Levina-Bickel failed: {e}")
            lb_dim = np.nan

        return MarketStructureResult(
            as_of_date=pd.to_datetime(as_of_date),
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            n_assets=prices.shape[1],
            n_observations=returns.shape[0],
            tickers=list(prices.columns),
            prices=prices,
            returns=returns,
            demeaned_returns=demeaned_returns,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            explained_variance_ratio=explained,
            cumulative_explained_variance=cumulative,
            rmt_result=rmt_result,
            participation_ratio_dimension=pr_dim,
            levina_bickel_dimension=lb_dim
        )

    # ------------------------------------------------------------------
    # Rolling window analysis
    # ------------------------------------------------------------------
    def rolling_dimension_analysis(
        self,
        as_of_date: str,
        full_start_date: str,
        full_end_date: str,
        window_size: int = 252,
        step_size: int = 21,
        return_method: str = "log",
        min_nonmissing_fraction: float = 0.80,
        max_assets: Optional[int] = None
    ) -> pd.DataFrame:
        tickers = self.historical_constituents(as_of_date)

        if max_assets is not None:
            tickers = tickers[:max_assets]

        prices = self.get_price_data(
            tickers=tickers,
            start_date=full_start_date,
            end_date=full_end_date,
            min_nonmissing_fraction=min_nonmissing_fraction
        )

        returns = self.compute_returns(prices, method=return_method)

        results = []
        dates = returns.index

        for start_idx in range(0, len(dates) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_returns = returns.iloc[start_idx:end_idx]

            demeaned_returns, covariance_matrix, _ = self.build_matrices(window_returns)
            eigenvalues, eigenvectors, explained, cumulative = self.eigendecomposition(covariance_matrix)

            rmt_result = self.rmt_dimension(eigenvalues, demeaned_returns)
            pr_dim = self.participation_ratio(eigenvalues)

            results.append({
                "window_start": dates[start_idx],
                "window_end": dates[end_idx - 1],
                "n_assets": covariance_matrix.shape[0],
                "n_observations": window_returns.shape[0],
                "rmt_dimension": rmt_result["dimension"],
                "participation_ratio_dimension": pr_dim,
                "top_eigenvalue": eigenvalues[0],
                "top_5_explained_variance": explained[:5].sum() if len(explained) >= 5 else explained.sum(),
                "mp_upper_bound": rmt_result["mp_upper_bound"]
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Eigenportfolio extraction
    # ------------------------------------------------------------------
    def eigenportfolio(
        self,
        result: MarketStructureResult,
        component: int = 0,
        normalize: str = "l1"
    ) -> pd.Series:
        """
        component=0 means top eigenvector.
        normalize:
            - 'l1': sum absolute weights = 1
            - 'l2': Euclidean norm = 1
            - 'sum1': weights sum to 1
            - None: raw eigenvector
        """
        if component < 0 or component >= result.eigenvectors.shape[1]:
            raise ValueError("Invalid component index.")

        vec = result.eigenvectors[:, component].copy()

        if normalize == "l1":
            denom = np.sum(np.abs(vec))
            if denom != 0:
                vec = vec / denom
        elif normalize == "l2":
            denom = np.linalg.norm(vec)
            if denom != 0:
                vec = vec / denom
        elif normalize == "sum1":
            denom = np.sum(vec)
            if denom != 0:
                vec = vec / denom
        elif normalize is None:
            pass
        else:
            raise ValueError("normalize must be one of {'l1','l2','sum1',None}.")

        return pd.Series(vec, index=result.covariance_matrix.index, name=f"PC{component+1}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_eigenvalue_spectrum(
        self,
        result: MarketStructureResult,
        show_rmt_threshold: bool = True,
        figsize: Tuple[int, int] = (12, 5)
    ):
        plt.figure(figsize=figsize)
        plt.plot(np.arange(1, len(result.eigenvalues) + 1), result.eigenvalues, marker="o")
        plt.xlabel("Eigenvalue Rank")
        plt.ylabel("Eigenvalue")
        plt.title("Covariance Eigenvalue Spectrum")

        if show_rmt_threshold:
            plt.axhline(
                result.rmt_result["mp_upper_bound"],
                linestyle="--",
                label="RMT Upper Bound"
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_explained_variance(
        self,
        result: MarketStructureResult,
        figsize: Tuple[int, int] = (12, 5)
    ):
        plt.figure(figsize=figsize)
        plt.plot(
            np.arange(1, len(result.cumulative_explained_variance) + 1),
            result.cumulative_explained_variance,
            marker="o"
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.tight_layout()
        plt.show()

    def plot_rolling_dimensions(
        self,
        rolling_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 5)
    ):
        plt.figure(figsize=figsize)
        plt.plot(rolling_df["window_end"], rolling_df["rmt_dimension"], label="RMT Dimension")
        plt.plot(
            rolling_df["window_end"],
            rolling_df["participation_ratio_dimension"],
            label="Participation Ratio Dimension"
        )
        plt.xlabel("Window End Date")
        plt.ylabel("Dimension")
        plt.title("Rolling Market Dimension")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summarize_result(
        self,
        result: MarketStructureResult
    ) -> pd.Series:
        return pd.Series({
            "as_of_date": result.as_of_date,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "n_assets": result.n_assets,
            "n_observations": result.n_observations,
            "top_eigenvalue": result.eigenvalues[0],
            "explained_var_pc1": result.explained_variance_ratio[0],
            "explained_var_top5": np.sum(result.explained_variance_ratio[:5]),
            "rmt_dimension": result.rmt_result["dimension"],
            "participation_ratio_dimension": result.participation_ratio_dimension,
            "levina_bickel_dimension": result.levina_bickel_dimension,
            "mp_upper_bound": result.rmt_result["mp_upper_bound"]
        })