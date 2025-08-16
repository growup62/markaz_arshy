import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from enum import Enum
import talib as ta

class TradingProfile(Enum):
    SCALPER = "scalper"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"

class ProfileAnalyzer:
    def __init__(self, config):
        self.config = config

    def _detect_bullish_bos(self, df):
        # Contoh deteksi BOS (placeholder)
        return df['close'].iloc[-1] > df['high'].iloc[-2]

    def _detect_fvg_bullish(self, df):
        # Contoh deteksi FVG (placeholder)
        return df['high'].iloc[-1] - df['low'].iloc[-2] > df['atr'].iloc[-1]

    def analyze_scalper(self, df):
        score = 0
        components = []
        if self._detect_bullish_bos(df):
            score += self.config["weights"].get("BULLISH_BOS", 0)
            components.append("BULLISH_BOS")
        if self._detect_fvg_bullish(df):
            score += self.config["weights"].get("FVG_BULLISH", 0)
            components.append("FVG_BULLISH")
        return {
            "score": score,
            "score_components": components,
            "profile_name": "scalping",
            "symbol": df["symbol"].iloc[0] if "symbol" in df else "unknown"
        }

    def analyze_intraday(self, df):
        score = 0
        components = []
        if self._detect_bullish_bos(df):
            score += self.config["weights"].get("BULLISH_BOS", 0)
            components.append("BULLISH_BOS")
        return {
            "score": score,
            "score_components": components,
            "profile_name": "intraday",
            "symbol": df["symbol"].iloc[0] if "symbol" in df else "unknown"
        }
    def _analyze_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisis khusus untuk position trader"""
        # Fokus pada analisis fundamental dan trend jangka panjang
        result = {
            'major_trend': self._analyze_major_trend(df),
            'structural_changes': self._detect_structural_changes(df),
            'volatility_regime': self._analyze_volatility_regime(df),
            'market_phase': self._determine_market_phase(df)
        }
        return result

    # --- Helper Methods untuk Scalper ---
    def _calculate_micro_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Hitung volatilitas mikro untuk scalping"""
        if len(df) < 20:
            return {'current': 0.0, 'average': 0.0}
        
        df['micro_vol'] = (df['high'] - df['low']) / df['low'] * 100
        current = float(df['micro_vol'].iloc[-1])
        average = float(df['micro_vol'].rolling(20).mean().iloc[-1])
        
        return {'current': current, 'average': average}
        
    def _calculate_short_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisis momentum jangka pendek"""
        if len(df) < 14:
            return {'rsi': 50.0, 'momentum': 0.0}
            
        # Hitung RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Hitung momentum
        momentum = df['close'].iloc[-1] - df['close'].iloc[-5]
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'momentum': float(momentum)
        }
        
    def _analyze_spread_impact(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisis dampak spread pada scalping"""
        if 'spread' not in df.columns or len(df) < 20:
            return {'spread_ratio': 1.0}
            
        avg_range = (df['high'] - df['low']).rolling(20).mean()
        spread_ratio = df['spread'] / avg_range
        
        return {'spread_ratio': float(spread_ratio.iloc[-1])}
        
    def _analyze_micro_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisis struktur mikro pasar"""
        if len(df) < 20:
            return {'pattern': 'unknown', 'strength': 0.0}
            
        # Deteksi pola-pola price action
        last_candles = df.tail(3)
        pattern = self._detect_candlestick_pattern(last_candles)
        strength = self._calculate_pattern_strength(last_candles)
        
        return {'pattern': pattern, 'strength': strength}

    # --- Helper Methods untuk Intraday ---
    def _calculate_day_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Hitung momentum harian"""
        if len(df) < 24:  # Minimal 24 candle untuk data harian
            return {'momentum': 0.0}
            
        day_open = df['open'].iloc[0]
        current_price = df['close'].iloc[-1]
        momentum = (current_price - day_open) / day_open * 100
        
        return {'momentum': float(momentum)}
        
    def _analyze_trading_session(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analisis sesi trading (Asia, London, New York)"""
        current_hour = pd.Timestamp.now().hour
        
        if 0 <= current_hour < 8:
            session = "ASIAN"
        elif 8 <= current_hour < 16:
            session = "LONDON"
        else:
            session = "NEW_YORK"
            
        return {'current_session': session}
        
    def _find_intraday_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Temukan level-level penting intraday"""
        if len(df) < 100:
            return {'support': [], 'resistance': []}
            
        # Identifikasi swing high/low
        highs = self._find_swing_points(df['high'], 'high')
        lows = self._find_swing_points(df['low'], 'low')
        
        return {
            'support': [float(x) for x in lows],
            'resistance': [float(x) for x in highs]
        }

    # --- Helper Methods untuk Swing Trading ---
    def _calculate_trend_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Hitung kekuatan trend untuk swing trading"""
        if len(df) < 50:
            return {'strength': 0.0}
            
        # Hitung ADX
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        dx = atr.diff() / atr * 100
        adx = abs(dx).rolling(14).mean()
        
        return {'strength': float(adx.iloc[-1])}
        
    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identifikasi level kunci untuk swing trading"""
        if len(df) < 100:
            return {'major_levels': []}
            
        # Temukan level-level yang sering ditest
        price_rounds = np.round(df['close'] / 10) * 10
        level_counts = price_rounds.value_counts()
        major_levels = level_counts[level_counts > level_counts.mean()].index.tolist()
        
        return {'major_levels': [float(x) for x in major_levels]}

    # --- Helper Methods untuk Position Trading ---
    def _analyze_major_trend(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analisis trend utama untuk position trading"""
        if len(df) < 200:
            return {'trend': 'UNDEFINED'}
            
        sma200 = df['close'].rolling(200).mean()
        current_price = df['close'].iloc[-1]
        
        if current_price > sma200.iloc[-1]:
            trend = 'BULLISH'
        else:
            trend = 'BEARISH'
            
        return {'trend': trend}
        
    def _detect_structural_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Deteksi perubahan struktur pasar jangka panjang"""
        if len(df) < 100:
            return {'change_detected': False}
            
        # Deteksi dengan moving average crossover
        ma50 = df['close'].rolling(50).mean()
        ma200 = df['close'].rolling(200).mean()
        
        cross_over = (ma50.iloc[-2] <= ma200.iloc[-2]) and (ma50.iloc[-1] > ma200.iloc[-1])
        cross_under = (ma50.iloc[-2] >= ma200.iloc[-2]) and (ma50.iloc[-1] < ma200.iloc[-1])
        
        return {
            'change_detected': cross_over or cross_under,
            'type': 'BULLISH' if cross_over else 'BEARISH' if cross_under else 'NONE'
        }

    def _find_swing_points(self, series: pd.Series, point_type: str, window: int = 5) -> List[float]:
        """Helper untuk menemukan swing points"""
        points = []
        for i in range(window, len(series) - window):
            if point_type == 'high':
                if all(series[i] > series[i-window:i]) and all(series[i] > series[i+1:i+window+1]):
                    points.append(series[i])
            else:  # low
                if all(series[i] < series[i-window:i]) and all(series[i] < series[i+1:i+window+1]):
                    points.append(series[i])
        return points[-3:] if points else []  # Return 3 terakhir saja

    def _detect_candlestick_pattern(self, candles: pd.DataFrame) -> str:
        """Helper untuk deteksi pola candlestick"""
        if len(candles) < 3:
            return "unknown"
            
        # Contoh sederhana: deteksi bullish engulfing
        prev_candle = candles.iloc[-2]
        curr_candle = candles.iloc[-1]
        
        if (prev_candle['close'] < prev_candle['open'] and  # Previous bearish
            curr_candle['close'] > curr_candle['open'] and  # Current bullish
            curr_candle['open'] < prev_candle['close'] and  # Opens below prev close
            curr_candle['close'] > prev_candle['open']):    # Closes above prev open
            return "bullish_engulfing"
            
        return "no_pattern"

    def _calculate_pattern_strength(self, candles: pd.DataFrame) -> float:
        """Helper untuk hitung kekuatan pola"""
        if len(candles) < 3:
            return 0.0
            
        # Hitung berdasarkan ukuran relatif candle dan volume
        curr_candle = candles.iloc[-1]
        body_size = abs(curr_candle['close'] - curr_candle['open'])
        avg_body = abs(candles['close'] - candles['open']).mean()
        
        strength = body_size / avg_body if avg_body != 0 else 1.0
        return min(strength, 1.0)  # Normalize to max 1.0

def calculate_slope(df: pd.DataFrame, length: int = 14, method: str = 'atr', multiplier: float = 1.0) -> float:
    """
    Menghitung slope untuk trendline berdasarkan beberapa metode.
    
    Args:
        df: DataFrame dengan data OHLCV
        length: Periode lookback
        method: Metode perhitungan ('atr', 'stdev', 'linreg')
        multiplier: Pengali slope
    
    Returns:
        float: Nilai slope
    """
    if method == 'atr':
        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(length).mean()
        return (atr.iloc[-1] / length) * multiplier
    
    elif method == 'stdev':
        return (df['close'].rolling(length).std().iloc[-1] / length) * multiplier
    
    elif method == 'linreg':
        # Linear regression slope
        x = np.arange(len(df[-length:]))
        y = df['close'].iloc[-length:].values
        slope, _ = np.polyfit(x, y, 1)
        return abs(slope) * multiplier
    
    return 0.0

def detect_pivot_points(df: pd.DataFrame, length: int = 14) -> Dict[str, Any]:
    """
    Mendeteksi pivot high dan pivot low.
    
    Args:
        df: DataFrame dengan data OHLCV
        length: Periode lookback
    
    Returns:
        Dict dengan pivot high dan low
    """
    highs = df['high'].rolling(2*length + 1, center=True).apply(
        lambda x: 1 if x.iloc[length] == max(x) else 0
    )
    lows = df['low'].rolling(2*length + 1, center=True).apply(
        lambda x: 1 if x.iloc[length] == min(x) else 0
    )
    
    pivot_high = df['high'].where(highs == 1)
    pivot_low = df['low'].where(lows == 1)
    
    return {
        'pivot_high': pivot_high.dropna(),
        'pivot_low': pivot_low.dropna()
    }

def analyze_trendlines(df: pd.DataFrame, length: int = 14, slope_mult: float = 1.0, 
                      calc_method: str = 'atr') -> Dict[str, Any]:
    """
    Menganalisis trendline dan deteksi breakout.
    
    Args:
        df: DataFrame dengan data OHLCV
        length: Periode untuk deteksi swing
        slope_mult: Pengali untuk perhitungan slope
        calc_method: Metode perhitungan slope ('atr', 'stdev', 'linreg')
    
    Returns:
        Dict dengan informasi trendline dan breakout
    """
    # Deteksi pivot points
    pivots = detect_pivot_points(df, length)
    
    # Hitung slope
    slope = calculate_slope(df, length, calc_method, slope_mult)
    
    # Inisialisasi variabel
    upper_trendline = None
    lower_trendline = None
    last_pivot_high = None
    last_pivot_low = None
    
    # Dapatkan pivot terakhir
    if len(pivots['pivot_high']) > 0:
        last_pivot_high = pivots['pivot_high'].iloc[-1]
    if len(pivots['pivot_low']) > 0:
        last_pivot_low = pivots['pivot_low'].iloc[-1]
    
    # Hitung trendlines
    current_price = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    
    if last_pivot_high is not None:
        bars_since_high = len(df) - pivots['pivot_high'].index[-1]
        upper_trendline = last_pivot_high - (slope * bars_since_high)
    
    if last_pivot_low is not None:
        bars_since_low = len(df) - pivots['pivot_low'].index[-1]
        lower_trendline = last_pivot_low + (slope * bars_since_low)
    
    # Deteksi breakout
    upward_break = False
    downward_break = False
    
    if upper_trendline is not None and current_high > upper_trendline:
        upward_break = True
    
    if lower_trendline is not None and current_low < lower_trendline:
        downward_break = True
    
    return {
        'upper_trendline': upper_trendline,
        'lower_trendline': lower_trendline,
        'upward_break': upward_break,
        'downward_break': downward_break,
        'slope': slope,
        'last_pivot_high': last_pivot_high,
        'last_pivot_low': last_pivot_low
    }

# --- Update existing get_market_context function ---
def get_market_context(df: pd.DataFrame, profile: Optional[str] = None) -> Dict[str, Any]:
    """Analisis lengkap konteks market dengan mempertimbangkan profil trading."""
    # Base context tetap sama
    context = {
        'zones': detect_premium_discount_zones(df),
        'structure': analyze_break_of_structure(df),
        'equilibrium': detect_market_equilibrium(df)
    }
    
    # Tambahkan analisis trendline
    trendline_analysis = analyze_trendlines(
        df, 
        length=14,  # Sesuaikan dengan kebutuhan
        slope_mult=1.0,
        calc_method='atr'
    )
    
    context['trendlines'] = trendline_analysis
    
    # Tentukan bias market dengan mempertimbangkan trendline breaks
    if trendline_analysis['upward_break']:
        context['structure']['trendline_break'] = 'BULLISH'
    elif trendline_analysis['downward_break']:
        context['structure']['trendline_break'] = 'BEARISH'
    else:
        context['structure']['trendline_break'] = None
    
    # Tambahkan analisis SMC zones dan profile-specific analysis
    smc_zones = calculate_smc_zones(df)
    zone_analysis = detect_zone_transition(df, smc_zones)
    
    context['smc_analysis'] = {
        'zones': smc_zones,
        'current_state': zone_analysis
    }
    
    if profile:
        try:
            profile_enum = TradingProfile(profile.lower())
            analyzer = ProfileAnalyzer(profile_enum)
            profile_analysis = analyzer.analyze(df)
            context['profile_analysis'] = profile_analysis
        except ValueError:
            context['profile_analysis'] = None
    
    # Tentukan bias market berdasarkan semua analisis
    if context['structure']['bos_detected']:
        bias = context['structure']['bos_direction']
    elif context['structure']['trendline_break']:
        bias = context['structure']['trendline_break']
    elif context['equilibrium']['equilibrium_detected']:
        bias = 'NEUTRAL'
    else:
        current_price = df['close'].iloc[-1]
        if current_price > context['zones']['premium']:
            bias = 'BULLISH'
        elif current_price < context['zones']['discount']:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
            
    context['market_bias'] = bias
    return context

def detect_premium_discount_zones(df: pd.DataFrame) -> Dict[str, float]:
    """Deteksi zona premium dan discount"""
    if len(df) < 20:
        return {'premium': df['high'].max(), 'discount': df['low'].min()}
        
    # Gunakan Volume Profile untuk menentukan zona
    volume_profile = analyze_volume_profile(df)
    high_vol_zones = sorted([zone for zone, vol in volume_profile.items() if vol > volume_profile.mean()], reverse=True)
    
    if len(high_vol_zones) >= 2:
        premium = high_vol_zones[0]
        discount = high_vol_zones[-1]
    else:
        # Fallback ke simple moving average
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        premium = sma20 * 1.01  # 1% di atas SMA
        discount = sma20 * 0.99  # 1% di bawah SMA
        
    return {'premium': float(premium), 'discount': float(discount)}

def analyze_break_of_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analisis Break of Structure (BOS)"""
    if len(df) < 50:
        return {'bos_detected': False, 'bos_direction': None, 'strength': 0.0}
        
    # Temukan swing high/low terakhir
    swing_highs = find_swing_points(df['high'], 'high', window=10)
    swing_lows = find_swing_points(df['low'], 'low', window=10)
    
    current_price = df['close'].iloc[-1]
    
    # Cek break of structure
    if swing_highs and current_price > max(swing_highs):
        return {
            'bos_detected': True,
            'bos_direction': 'BULLISH',
            'strength': (current_price - max(swing_highs)) / df['atr'].iloc[-1]
        }
    elif swing_lows and current_price < min(swing_lows):
        return {
            'bos_detected': True,
            'bos_direction': 'BEARISH',
            'strength': (min(swing_lows) - current_price) / df['atr'].iloc[-1]
        }
        
    return {'bos_detected': False, 'bos_direction': None, 'strength': 0.0}

def detect_market_equilibrium(df: pd.DataFrame) -> Dict[str, Any]:
    """Deteksi keseimbangan market (equilibrium)"""
    if len(df) < 20:
        return {'equilibrium_detected': False, 'confidence': 0.0}
        
    # Hitung volatilitas relatif
    atr = df['atr'].iloc[-1]
    recent_range = df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()
    
    # Hitung volume relatif
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    
    # Deteksi kondisi equilibrium
    low_volatility = recent_range < atr * 0.5
    declining_volume = current_volume < avg_volume * 0.7
    
    equilibrium_detected = low_volatility and declining_volume
    
    # Hitung tingkat keyakinan
    if equilibrium_detected:
        volatility_ratio = 1 - (recent_range / (atr * 0.5))
        volume_ratio = 1 - (current_volume / (avg_volume * 0.7))
        confidence = (volatility_ratio + volume_ratio) / 2
    else:
        confidence = 0.0
        
    return {
        'equilibrium_detected': equilibrium_detected,
        'confidence': float(confidence)
    }

def analyze_volume_profile(df: pd.DataFrame) -> Dict[float, float]:
    """Analisis Volume Profile untuk menentukan area interest"""
    if len(df) < 100:
        return {}
        
    # Buat price bins
    price_range = df['high'].max() - df['low'].min()
    bin_size = price_range / 20  # 20 bins
    
    # Hitung volume untuk setiap price level
    price_points = (df['high'] + df['low']) / 2  # Point of Control
    volume_profile = {}
    
    for price in np.arange(df['low'].min(), df['high'].max(), bin_size):
        mask = (price_points >= price) & (price_points < price + bin_size)
        volume_profile[float(price)] = float(df.loc[mask, 'volume'].sum())
        
    return volume_profile

def detect_smc_structure(df: pd.DataFrame) -> Dict[str, Any]:
    # Placeholder for SMC structure detection. This function should be implemented or imported from technical_indicators_smc.py
    return {'internal_structure': {}, 'swing_structure': {}, 'order_blocks': [], 'fair_value_gaps': [], 'equal_levels': {'highs': [], 'lows': []}}

def analyze_market_structure(df: pd.DataFrame, predictor=None, history_analyzer=None) -> Dict[str, Any]:
    """Enhanced market structure analysis with prediction and historical insights"""
    
    # Get current market analysis
    current_analysis = detect_smc_structure(df)
    
    # Add future price prediction if predictor is available
    if predictor:
        prediction = predictor.predict(df)
        current_analysis['future_prediction'] = prediction
        
        # Add trading bias based on prediction
        if prediction['confidence'] > 0.6:
            if prediction['predicted_price'] > prediction['current_price']:
                current_analysis['trading_bias'] = 'BULLISH'
            else:
                current_analysis['trading_bias'] = 'BEARISH'
        else:
            current_analysis['trading_bias'] = 'NEUTRAL'
    
    # Add historical insights if analyzer is available
    if history_analyzer and hasattr(history_analyzer, 'profitable_patterns'):
        historical_insights = {
            'profitable_patterns': history_analyzer.profitable_patterns,
            'optimal_conditions': history_analyzer.market_conditions
        }
        current_analysis['historical_insights'] = historical_insights
        
        # Check if current conditions match profitable historical patterns
        current_conditions = {
            'volatility': df['high'].rolling(20).std().iloc[-1],
            'volume': df['volume'].iloc[-1] if 'volume' in df.columns else None,
            'hour': pd.to_datetime(df['time'].iloc[-1]).hour
        }
        
        pattern_match_score = _calculate_pattern_match(
            current_conditions,
            historical_insights['optimal_conditions']
        )
        current_analysis['pattern_match_score'] = pattern_match_score
        
    return current_analysis

def _calculate_pattern_match(current: Dict[str, Any], optimal: Dict[str, Any]) -> float:
    """Calculate how well current conditions match historical optimal conditions"""
    scores = []
    
    # Check volatility match
    if 'volatility' in current and 'volatility_range' in optimal:
        vol_range = optimal['volatility_range']
        if vol_range['min'] <= current['volatility'] <= vol_range['max']:
            scores.append(1.0)
        else:
            distance = min(abs(current['volatility'] - vol_range['min']),
                         abs(current['volatility'] - vol_range['max']))
            scores.append(max(0, 1 - distance/vol_range['optimal']))
            
    # Check volume match
    if (current['volume'] is not None and 'volume_range' in optimal):
        vol_range = optimal['volume_range']
        if vol_range['min'] <= current['volume'] <= vol_range['max']:
            scores.append(1.0)
        else:
            distance = min(abs(current['volume'] - vol_range['min']),
                         abs(current['volume'] - vol_range['max']))
            scores.append(max(0, 1 - distance/vol_range['optimal']))
            
    # Check trading hour match
    if current['hour'] in optimal.get('best_hours', []):
        scores.append(1.0)
    else:
        scores.append(0.0)
        
    return sum(scores) / len(scores) if scores else 0.0

class MarketPredictor:
    def __init__(self, input_size=5):
        self.model = TradingPredictor(input_size)  # Dari class di atas

    def train(self, df):
        # Extract features: misal _prepare_features(df)
        features = np.array([self._prepare_features(df.iloc[i]) for i in range(len(df))])  # Adjust
        labels = (df['close'].shift(-1) > df['close']).astype(int)[:-1]  # Simple label: 1 if up
        features = features[:-1]
        self.model = train_predictor(features, labels, epochs=50, lr=0.001)

    def predict(self, df):
        features = self._prepare_features(df.iloc[-1])
        features = torch.FloatTensor([features])
        pred = self.model(features)
        return {'buy_signal': pred[0][1].item(), 'sell_signal': pred[0][0].item()}
        
        # Proses prediksi
        features = self._prepare_features(df)
        prediction = self.model.predict(features)
        
        return {
            'buy_signal': float(prediction[0]),
            'sell_signal': float(prediction[1])
        }
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Siapkan fitur untuk model prediksi"""
        # Contoh: gunakan hanya harga penutupan untuk regresi linier sederhana
        return df['close'].values.reshape(-1, 1)

class HistoryAnalyzer:
    def __init__(self):
        pass
    
    def analyze_historical_trades(self, df: pd.DataFrame, past_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisis trade historis untuk menemukan pola dan evaluasi performa"""
        # Contoh analisis: hitung rasio profitabilitas
        if not past_trades:
            return {'profit_ratio': 0.0, 'loss_ratio': 0.0}
        
        total_profit = sum(trade['profit'] for trade in past_trades)
        total_loss = sum(trade['loss'] for trade in past_trades)
        
        profit_ratio = total_profit / (total_profit + total_loss) if (total_profit + total_loss) != 0 else 0
        
        return {
            'profit_ratio': profit_ratio,
            'loss_ratio': 1 - profit_ratio
        }

predictor = MarketPredictor()
analyzer = HistoryAnalyzer()

