# === UTILITY FUNCTIONS FOR ANALYZE_MARKET ===

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Hitung Average True Range (ATR)"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def find_swing_points(series: pd.Series, point_type: str, window: int = 5) -> List[float]:
    """Helper untuk menemukan swing points"""
    points = []
    for i in range(window, len(series) - window):
        if point_type == 'high':
            if all(series.iloc[i] > series.iloc[i-window:i]) and all(series.iloc[i] > series.iloc[i+1:i+window+1]):
                points.append(series.iloc[i])
        else:  # low
            if all(series.iloc[i] < series.iloc[i-window:i]) and all(series.iloc[i] < series.iloc[i+1:i+window+1]):
                points.append(series.iloc[i])
    return points[-3:] if points else []  # Return 3 terakhir saja

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Persiapkan dataframe dengan indikator yang diperlukan"""
    df = df.copy()
    
    # Tambahkan ATR jika belum ada
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)
    
    # Tambahkan volume jika belum ada (untuk testing)
    if 'volume' not in df.columns:
        df['volume'] = 1000  # Default volume
        
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Hitung indikator teknikal utama"""
    indicators = {}
    
    # RSI
    if len(df) >= 14:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = float(rsi.iloc[-1])
    else:
        indicators['rsi'] = 50.0
    
    # MACD
    if len(df) >= 26:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        indicators['macd'] = {
            'macd': float(macd.iloc[-1]),
            'signal': float(signal.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    else:
        indicators['macd'] = {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    # Bollinger Bands
    if len(df) >= 20:
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        
        indicators['bollinger_bands'] = {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma20.iloc[-1]),
            'lower': float(lower_band.iloc[-1]),
            'position': 'upper' if df['close'].iloc[-1] > upper_band.iloc[-1] else 
                       'lower' if df['close'].iloc[-1] < lower_band.iloc[-1] else 'middle'
        }
    else:
        current_price = df['close'].iloc[-1]
        indicators['bollinger_bands'] = {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98,
            'position': 'middle'
        }
    
    return indicators

def analyze_market_sentiment(df: pd.DataFrame) -> Dict[str, Any]:
    """Analisis sentimen market"""
    sentiment = {}
    
    # Volume trend
    if 'volume' in df.columns and len(df) >= 10:
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-20:].mean() if len(df) >= 20 else df['volume'].mean()
        
        if recent_volume > avg_volume * 1.2:
            volume_sentiment = 'high'
        elif recent_volume < avg_volume * 0.8:
            volume_sentiment = 'low'
        else:
            volume_sentiment = 'normal'
            
        sentiment['volume_sentiment'] = volume_sentiment
    else:
        sentiment['volume_sentiment'] = 'normal'
    
    # Price momentum
    if len(df) >= 5:
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
        
        if price_change > 2:
            price_sentiment = 'bullish'
        elif price_change < -2:
            price_sentiment = 'bearish'
        else:
            price_sentiment = 'neutral'
            
        sentiment['price_sentiment'] = price_sentiment
        sentiment['price_change_5d'] = float(price_change)
    else:
        sentiment['price_sentiment'] = 'neutral'
        sentiment['price_change_5d'] = 0.0
    
    return sentiment

def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Hitung metrik risiko"""
    risk_metrics = {}
    
    # Volatilitas
    if len(df) >= 20:
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        risk_metrics['volatility'] = float(volatility.iloc[-1])
    else:
        risk_metrics['volatility'] = 0.2  # Default 20%
    
    # Maximum Drawdown
    if len(df) >= 10:
        cumulative = (1 + np.log(df['close'] / df['close'].shift(1)).fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        risk_metrics['max_drawdown'] = float(max_drawdown)
    else:
        risk_metrics['max_drawdown'] = 0.0
    
    # Sharpe Ratio (simplified)
    if len(df) >= 30:
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        if returns.std() != 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            risk_metrics['sharpe_ratio'] = float(sharpe)
        else:
            risk_metrics['sharpe_ratio'] = 0.0
    else:
        risk_metrics['sharpe_ratio'] = 0.0
    
    return risk_metrics