"""
technical_indicators.py (refactor advanced)
===========================================

Versi refaktor: multi-zone detection, scoring, pattern, boundary, pivot,
feature extractor, zone plotting & event logging.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging  # Tambah logging untuk robustness

def calculate_ma(df: pd.DataFrame, period: int = 50) -> Optional[float]:
    """
    Menghitung nilai Simple Moving Average (MA).
    """
    if 'close' not in df.columns or df.empty or len(df) < period:
        logging.warning("Data tidak cukup atau kolom 'close' tidak ada untuk calculate_ma.")
        return None

    ma_value = df['close'].rolling(window=period).mean().iloc[-1]
    return float(ma_value) if pd.notna(ma_value) else None

# ========== ZONE & STRUCTURE ==========

def calculate_atr_dynamic(df: pd.DataFrame, period: int = 14) -> float:
    if df.empty or len(df) < period + 1:
        return 0.0
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(atr_val) if pd.notna(atr_val) else 0.0

def detect_structure(df: pd.DataFrame, swing_lookback: int = 5) -> Tuple[str, Dict[str, Any]]:
    if df.empty or len(df) < swing_lookback * 2:
        return "INDECISIVE", {}

    df = df.copy()

    # Deteksi Swing Highs and Lows yang lebih robust
    df['is_swing_high'] = (df['high'] == df['high'].rolling(swing_lookback * 2 + 1, center=True).max())
    df['is_swing_low'] = (df['low'] == df['low'].rolling(swing_lookback * 2 + 1, center=True).min())

    swing_highs = df[df['is_swing_high']]
    swing_lows = df[df['is_swing_low']]

    res = []
    last_swing_high_price = None
    last_swing_low_price = None
    last_swing_high_idx = None
    last_swing_low_idx = None

    # Urutkan swing points berdasarkan waktu (index)
    if not swing_highs.empty:
        last_sh = swing_highs.iloc[-1]
        last_swing_high_price = last_sh['high']
        last_swing_high_idx = last_sh.name
        if len(swing_highs) > 1:
            prev_sh = swing_highs.iloc[-2]
            if last_sh['high'] > prev_sh['high']:
                res.append('HH')
            else:
                res.append('LH')

    if not swing_lows.empty:
        last_sl = swing_lows.iloc[-1]
        last_swing_low_price = last_sl['low']
        last_swing_low_idx = last_sl.name
        if len(swing_lows) > 1:
            prev_sl = swing_lows.iloc[-2]
            if last_sl['low'] > prev_sl['low']:
                res.append('HL')
            else:
                res.append('LL')

    # --- LOGIKA BARU UNTUK BREAK OF STRUCTURE (BOS) ---
    # Membutuhkan penutupan candle (candle close) untuk konfirmasi
    if last_swing_high_idx is not None:
        # Cek candle SETELAH swing high terakhir
        subsequent_candles = df.loc[last_swing_high_idx:].iloc[1:]
        confirmed_break = subsequent_candles[subsequent_candles['close'] > last_swing_high_price]
        if not confirmed_break.empty:
            res.append("BULLISH_BOS")

    if last_swing_low_idx is not None:
        # Cek candle SETELAH swing low terakhir
        subsequent_candles = df.loc[last_swing_low_idx:].iloc[1:]
        confirmed_break = subsequent_candles[subsequent_candles['close'] < last_swing_low_price]
        if not confirmed_break.empty:
            res.append("BEARISH_BOS")

    structure_str = "/".join(sorted(list(set(res)))) if res else "INDECISIVE"
    return structure_str, {'last_high': last_swing_high_price, 'last_low': last_swing_low_price}

def detect_order_blocks_multi(
    df: pd.DataFrame, lookback: int = 15, structure_filter: Optional[str] = None, max_age: int = 20
) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    order_blocks = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(len(df) - lookback - 2, len(df) - 2):
        candle = df.iloc[i]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish OB
        if candle['close'] < candle['open']:
            found_bos = False
            for j in range(i + 1, min(i + lookback + 1, len(df))):
                if df.iloc[j]['close'] > candle['high']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BULLISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open'] + candle['close']) / 2)),
                })
        # Bearish OB
        elif candle['close'] > candle['open']:
            found_bos = False
            for j in range(i + 1, min(i + lookback + 1, len(df))):
                if df.iloc[j]['close'] < candle['low']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BEARISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BEARISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open'] + candle['close']) / 2)),
                })
    order_blocks = sorted(order_blocks, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return order_blocks

def detect_fvg_multi(df: pd.DataFrame, min_gap: float = 0.0002, max_age: int = 20) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    fvg_zones = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(1, len(df) - 1):
        c_prev, c_now, c_next = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish FVG
        gap = c_next['low'] - c_prev['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BULLISH',
                'start': float(c_prev['high']),
                'end': float(c_next['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_prev['high'] + c_next['low']) / 2))
            })
        # Bearish FVG
        gap = c_prev['low'] - c_next['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BEARISH',
                'start': float(c_next['high']),
                'end': float(c_prev['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_next['high'] + c_prev['low']) / 2))
            })
    fvg_zones = sorted(fvg_zones, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return fvg_zones

# ========== PATTERN, VOLUME, BOUNDARY, PIVOT ==========

def detect_pinbar(df: pd.DataFrame, min_ratio: float = 2.0) -> List[Dict[str, Any]]:
    """Pinbar: ekor minimal 2x body."""
    if df.empty or len(df) < 3:
        return []
    pattern = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        body = abs(c['close'] - c['open'])
        upper = c['high'] - max(c['close'], c['open'])
        lower = min(c['close'], c['open']) - c['low']
        # Bullish pinbar
        if lower > min_ratio * body and upper < 0.5 * body:
            pattern.append({'type': 'PINBAR_BULL', 'time': c['time'], 'idx': i})
        # Bearish pinbar
        elif upper > min_ratio * body and lower < 0.5 * body:
            pattern.append({'type': 'PINBAR_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def detect_engulfing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty or len(df) < 2:
        return []
    pattern = []
    for i in range(1, len(df)):
        prev, c = df.iloc[i - 1], df.iloc[i]
        # Bullish engulfing
        if c['close'] > c['open'] and prev['close'] < prev['open'] and \
           c['close'] > prev['open'] and c['open'] < prev['close']:
            pattern.append({'type': 'ENGULFING_BULL', 'time': c['time'], 'idx': i})
        # Bearish engulfing
        elif c['close'] < c['open'] and prev['close'] > prev['open'] and \
             c['close'] < prev['open'] and c['open'] > prev['close']:
            pattern.append({'type': 'ENGULFING_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for momentum check."""
    if series.empty or len(series) < period + 1:
        return pd.Series()
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Default to 50 if NaN

def analyze_tf_opportunity(df: pd.DataFrame, symbol: str, tf: str, mt5_path: str, start_date: str, end_date: str, profile_type: str, weights: Dict[str, float]) -> Dict[str, Any]:
    """Fungsi analisis utama: Gabungkan structure, OB, FVG, patterns untuk scoring."""
    if df.empty:
        return {}
    
    structure, details = detect_structure(df)
    order_blocks = detect_order_blocks_multi(df)
    fvgs = detect_fvg_multi(df)
    pinbars = detect_pinbar(df)
    engulfings = detect_engulfing(df)
    patterns = pinbars + engulfings
    
    score = 0.0
    info_list = []
    
    # Scoring dari structure
    if 'BULLISH_BOS' in structure:
        score += weights.get('BOS_BULLISH', 5.0)
        info_list.append('Bullish BOS')
    elif 'BEARISH_BOS' in structure:
        score -= weights.get('BOS_BEARISH', 5.0)
        info_list.append('Bearish BOS')
    
    # Scoring dari OB
    for ob in order_blocks:
        if ob['type'] == 'BULLISH_OB':
            score += weights.get('OB_BULLISH', 3.0) * ob['strength']
            info_list.append('Bullish OB')
        elif ob['type'] == 'BEARISH_OB':
            score -= weights.get('OB_BEARISH', 3.0) * ob['strength']
            info_list.append('Bearish OB')
    
    # Scoring dari FVG
    for fvg in fvgs:
        if fvg['type'] == 'FVG_BULLISH':
            score += weights.get('FVG_BULLISH', 2.0) * fvg['strength']
            info_list.append('Bullish FVG')
        elif fvg['type'] == 'FVG_BEARISH':
            score -= weights.get('FVG_BEARISH', 2.0) * fvg['strength']
            info_list.append('Bearish FVG')
    
    # Scoring dari patterns
    for pattern in patterns:
        if 'BULL' in pattern['type']:
            score += weights.get('PATTERN_BULL', 1.5)
            info_list.append(pattern['type'])
        elif 'BEAR' in pattern['type']:
            score -= weights.get('PATTERN_BEAR', 1.5)
            info_list.append(pattern['type'])
    
    return {
        'score': score,
        'info_list': list(set(info_list)),  # Hindari duplikat
        'structure': structure,
        'order_blocks': order_blocks,
        'fair_value_gaps': fvgs,
        'patterns': patterns
    }

def adapt_weights_to_context(weights: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
    """
    Adaptasi bobot berdasarkan kondisi market.
    """
    adapted = weights.copy()

    # Sesuaikan berdasarkan fase trend
    if context.get('trend_phase') == 'RANGING':
        for key in adapted:
            if 'BREAKOUT' in key or 'BOS' in key:
                adapted[key] *= 0.7
            if 'REVERSAL' in key or 'CHoCH' in key:
                adapted[key] *= 1.3
    elif context.get('trend_phase') == 'TRENDING':
        for key in adapted:
            if 'BREAKOUT' in key or 'BOS' in key or 'CONTINUATION' in key:
                adapted[key] *= 1.3
            if 'REVERSAL' in key or 'CHoCH' in key:
                adapted[key] *= 0.7

    # Sesuaikan berdasarkan volatilitas
    vol_ratio = context.get('volatility', {}).get('volatility_ratio', 1.0)
    if vol_ratio > 1.5:  # High volatility
        for key in adapted:
            if 'PATTERN' in key or 'ENGULFING' in key or 'PINBAR' in key:
                adapted[key] *= 0.8
            if 'STRUCTURE' in key or 'BOS' in key:
                adapted[key] *= 1.2
    elif vol_ratio < 0.5:  # Low volatility
        for key in adapted:
            if 'PATTERN' in key or 'ENGULFING' in key or 'PINBAR' in key:
                adapted[key] *= 1.2

    # Sesuaikan berdasarkan kondisi likuiditas
    liquidity = context.get('liquidity_conditions', {})
    if liquidity.get('average_spread', 0) > 2.0:  # Spread tinggi
        for key in adapted:
            adapted[key] *= 0.8

    return adapted

def validate_scalping_opportunity(
    df: pd.DataFrame, profile_settings: Dict[str, Any]
) -> Tuple[bool, float, List[str]]:
    """
    Validasi khusus untuk scalping dengan kriteria yang lebih ketat
    """
    if df.empty:
        return False, 0, ["Data kosong untuk validasi scalping."]
    
    validations = []
    confidence_boost = 0

    # 1. Analisis Volume
    volume_ma = df['tick_volume'].rolling(20).mean()
    current_volume = df['tick_volume'].iloc[-1]
    volume_ratio = current_volume / volume_ma.iloc[-1] if pd.notna(volume_ma.iloc[-1]) else 0

    if volume_ratio >= profile_settings.get('entry_rules', {}).get('minimum_volume_threshold', 1.5):
        validations.append(f"Volume valid ({volume_ratio:.2f}x average)")
        confidence_boost += 0.5
    else:
        return False, 0, ["Volume tidak mencukupi"]

    # 2. Analisis Volatilitas
    volatility_percentile = df['high'].sub(df['low']).rolling(50).mean().rank(pct=True).iloc[-1] * 100 if not df.empty else 0

    min_vol = profile_settings.get('entry_rules', {}).get('minimum_volatility_percentile', 30)
    max_vol = profile_settings.get('entry_rules', {}).get('maximum_volatility_percentile', 70)
    if min_vol <= volatility_percentile <= max_vol:
        validations.append(f"Volatilitas optimal ({volatility_percentile:.1f}%)")
        confidence_boost += 0.5
    else:
        return False, 0, ["Volatilitas di luar range optimal"]

    # 3. Momentum Check
    rsi = calculate_rsi(df['close'])
    last_rsi = rsi.iloc[-1] if not rsi.empty else 50

    if 20 <= last_rsi <= 80:  # Avoid extreme conditions
        momentum_aligned = (last_rsi > 50 and df['close'].iloc[-1] > df['close'].iloc[-2]) or \
                           (last_rsi < 50 and df['close'].iloc[-1] < df['close'].iloc[-2])
        if momentum_aligned:
            validations.append("Momentum aligned")
            confidence_boost += 0.7

    # 4. Pattern Quality Check
    pattern_quality_threshold = profile_settings.get('advanced_filters', {}).get('pattern_quality_threshold', 0.5)
    if pattern_quality_threshold:
        pattern_quality = analyze_pattern_quality(df)
        if pattern_quality >= pattern_quality_threshold:
            validations.append(f"Pattern quality good ({pattern_quality:.2f})")
            confidence_boost += 0.8
        else:
            return False, 0, ["Pattern quality insufficient"]

    # 5. Price Action Clarity
    price_clarity = analyze_price_action_clarity(df)
    if price_clarity >= 0.7:
        validations.append(f"Clear price action ({price_clarity:.2f})")
        confidence_boost += 0.5

    # 6. Reward Ratio Check
    min_reward_ratio = profile_settings.get('advanced_filters', {}).get('minimum_reward_ratio', 1.5)
    if min_reward_ratio:
        nearest_resistance = find_nearest_resistance(df)
        nearest_support = find_nearest_support(df)
        current_price = df['close'].iloc[-1]

        potential_reward = abs(nearest_resistance - current_price)
        potential_risk = abs(current_price - nearest_support)

        if potential_risk > 0:
            reward_ratio = potential_reward / potential_risk
            if reward_ratio >= min_reward_ratio:
                validations.append(f"Good reward ratio ({reward_ratio:.2f})")
                confidence_boost += 0.6
            else:
                return False, 0, ["Insufficient reward ratio"]

    return len(validations) >= 4, confidence_boost, validations

def analyze_pattern_quality(df: pd.DataFrame) -> float:
    """Analisis kualitas pattern untuk scalping"""
    if df.empty or len(df) < 2:
        return 0.0
    
    quality_score = 0.0
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # 1. Body to Wick Ratio
    body = abs(last_candle['close'] - last_candle['open'])
    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']

    if body > (upper_wick + lower_wick):
        quality_score += 0.4  # Strong conviction

    # 2. Clean Break of Previous Structure
    if last_candle['close'] > prev_candle['high'] or last_candle['close'] < prev_candle['low']:
        quality_score += 0.3

    # 3. Volume Confirmation
    if 'tick_volume' in df.columns and last_candle['tick_volume'] > df['tick_volume'].rolling(20).mean().iloc[-1]:
        quality_score += 0.3

    return min(quality_score, 1.0)

def analyze_price_action_clarity(df: pd.DataFrame) -> float:
    """Analisis kejelasan price action untuk scalping"""
    if df.empty or len(df) < 5:
        return 0.0
    
    clarity_score = 0.0
    recent_candles = df.tail(5)

    # 1. Consistency in Direction
    closes = recent_candles['close']
    diffs = closes.diff().dropna()
    if (diffs > 0).all() or (diffs < 0).all():
        clarity_score += 0.4

    # 2. Clean Candle Formation
    for _, candle in recent_candles.iterrows():
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        if total_range > 0 and body / total_range > 0.6:
            clarity_score += 0.1  # Max 0.5 from this

    # 3. Minimal Overlap
    for i in range(1, len(recent_candles)):
        curr = recent_candles.iloc[i]
        prev = recent_candles.iloc[i - 1]
        if (curr['high'] > prev['high'] and curr['low'] > prev['low']) or \
           (curr['high'] < prev['high'] and curr['low'] < prev['low']):
            clarity_score += 0.1  # Max 0.4 from this

    return min(clarity_score, 1.0)

def find_nearest_support(df: pd.DataFrame) -> float:
    """Mencari level support terdekat"""
    if df.empty:
        return 0.0
    
    recent_lows = df['low'].rolling(window=20).min()
    current_price = df['close'].iloc[-1]

    support_levels = recent_lows[recent_lows < current_price].unique()
    if len(support_levels) > 0:
        return max(support_levels)  # Highest support below current price
    return current_price - calculate_atr_dynamic(df)

def find_nearest_resistance(df: pd.DataFrame) -> float:
    """Mencari level resistance terdekat"""
    if df.empty:
        return 0.0
    
    recent_highs = df['high'].rolling(window=20).max()
    current_price = df['close'].iloc[-1]

    resistance_levels = recent_highs[recent_highs > current_price].unique()
    if len(resistance_levels) > 0:
        return min(resistance_levels)  # Lowest resistance above current price
    return current_price + calculate_atr_dynamic(df)