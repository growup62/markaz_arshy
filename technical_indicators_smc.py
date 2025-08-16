# technical_indicators_smc.py
# Pure Smart Money Concepts (SMC) Analysis - No Traditional Indicators
# Focus on Market Structure, Liquidity, and Institutional Behavior

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import lru_cache
from datetime import datetime, timedelta
from enum import Enum

# =========================
# SMC Enums & Constants
# =========================

class MarketStructure(Enum):
    BOS_BULLISH = "BOS_BULLISH"  # Break of Structure Bullish
    BOS_BEARISH = "BOS_BEARISH"  # Break of Structure Bearish
    CHOCH_BULLISH = "CHOCH_BULLISH"  # Change of Character Bullish
    CHOCH_BEARISH = "CHOCH_BEARISH"  # Change of Character Bearish
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low
    RANGE = "RANGE"  # Ranging Market

class LiquidityType(Enum):
    BSL = "BSL"  # Buy Side Liquidity
    SSL = "SSL"  # Sell Side Liquidity
    EQH = "EQH"  # Equal Highs
    EQL = "EQL"  # Equal Lows
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"

class OrderBlockType(Enum):
    BULLISH_OB = "BULLISH_OB"  # Bullish Order Block
    BEARISH_OB = "BEARISH_OB"  # Bearish Order Block
    MITIGATION_BLOCK = "MITIGATION_BLOCK"  # Mitigated Order Block
    BREAKER_BLOCK = "BREAKER_BLOCK"  # Breaker Block

class FVGType(Enum):
    BULLISH_FVG = "BULLISH_FVG"  # Bullish Fair Value Gap
    BEARISH_FVG = "BEARISH_FVG"  # Bearish Fair Value Gap
    BALANCED_PRICE_RANGE = "BALANCED_PRICE_RANGE"  # Balanced Price Range

# =========================
# Core SMC Utilities
# =========================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range for volatility measurement"""
    if len(df) < period + 1:
        return 0.0
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]

def identify_swing_points(df: pd.DataFrame, swing_length: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify swing highs and lows with improved accuracy"""
    if len(df) < swing_length * 2 + 1:
        return pd.DataFrame(), pd.DataFrame()
    
    df = df.copy()
    
    # Enhanced swing detection with confirmation
    swing_high_condition = (
        (df['high'].rolling(window=swing_length*2+1, center=True).max() == df['high']) &
        (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
    )
    
    swing_low_condition = (
        (df['low'].rolling(window=swing_length*2+1, center=True).min() == df['low']) &
        (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
    )
    
    swing_highs = df[swing_high_condition].copy()
    swing_lows = df[swing_low_condition].copy()
    
    # Add additional context
    if not swing_highs.empty:
        swing_highs['strength'] = swing_highs['high'] - swing_highs[['open', 'close']].min(axis=1)
    
    if not swing_lows.empty:
        swing_lows['strength'] = swing_lows[['open', 'close']].max(axis=1) - swing_lows['low']
    
    return swing_highs, swing_lows

# =========================
# Market Structure Analysis
# =========================

def analyze_market_structure(df: pd.DataFrame, swing_length: int = 5) -> Dict[str, Any]:
    """Comprehensive market structure analysis using pure SMC concepts"""
    if df.empty or len(df) < swing_length * 2 + 1:
        return {
            'structure_type': MarketStructure.RANGE.value,
            'trend_direction': 'NEUTRAL',
            'bos_detected': False,
            'choch_detected': False,
            'last_swing_high': None,
            'last_swing_low': None,
            'structure_strength': 0.0
        }
    
    swing_highs, swing_lows = identify_swing_points(df, swing_length)
    
    if swing_highs.empty or swing_lows.empty:
        return {
            'structure_type': MarketStructure.RANGE.value,
            'trend_direction': 'NEUTRAL',
            'bos_detected': False,
            'choch_detected': False,
            'last_swing_high': None,
            'last_swing_low': None,
            'structure_strength': 0.0
        }
    
    # Get recent swings
    recent_highs = swing_highs.tail(3)
    recent_lows = swing_lows.tail(3)
    
    structure_signals = []
    
    # Analyze Higher Highs and Higher Lows (Bullish Structure)
    if len(recent_highs) >= 2:
        if recent_highs.iloc[-1]['high'] > recent_highs.iloc[-2]['high']:
            structure_signals.append('HH')
    
    if len(recent_lows) >= 2:
        if recent_lows.iloc[-1]['low'] > recent_lows.iloc[-2]['low']:
            structure_signals.append('HL')
    
    # Analyze Lower Highs and Lower Lows (Bearish Structure)
    if len(recent_highs) >= 2:
        if recent_highs.iloc[-1]['high'] < recent_highs.iloc[-2]['high']:
            structure_signals.append('LH')
    
    if len(recent_lows) >= 2:
        if recent_lows.iloc[-1]['low'] < recent_lows.iloc[-2]['low']:
            structure_signals.append('LL')
    
    # Determine overall structure
    if 'HH' in structure_signals and 'HL' in structure_signals:
        structure_type = MarketStructure.BOS_BULLISH.value
        trend_direction = 'BULLISH'
    elif 'LH' in structure_signals and 'LL' in structure_signals:
        structure_type = MarketStructure.BOS_BEARISH.value
        trend_direction = 'BEARISH'
    else:
        structure_type = MarketStructure.RANGE.value
        trend_direction = 'NEUTRAL'
    
    # Check for BOS (Break of Structure)
    current_price = df['close'].iloc[-1]
    bos_detected = False
    choch_detected = False
    
    if not swing_highs.empty:
        last_significant_high = swing_highs['high'].max()
        if current_price > last_significant_high:
            bos_detected = True
            structure_type = MarketStructure.BOS_BULLISH.value
    
    if not swing_lows.empty:
        last_significant_low = swing_lows['low'].min()
        if current_price < last_significant_low:
            bos_detected = True
            structure_type = MarketStructure.BOS_BEARISH.value
    
    # Calculate structure strength
    structure_strength = len([s for s in structure_signals if s in ['HH', 'HL', 'LH', 'LL']]) / 4.0
    
    return {
        'structure_type': structure_type,
        'trend_direction': trend_direction,
        'bos_detected': bos_detected,
        'choch_detected': choch_detected,
        'last_swing_high': swing_highs['high'].iloc[-1] if not swing_highs.empty else None,
        'last_swing_low': swing_lows['low'].iloc[-1] if not swing_lows.empty else None,
        'structure_strength': structure_strength,
        'structure_signals': structure_signals
    }

# =========================
# Liquidity Analysis
# =========================

def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
    """Detect liquidity zones (BSL/SSL) where stops are likely placed"""
    if df.empty:
        return []
    
    swing_highs, swing_lows = identify_swing_points(df)
    liquidity_zones = []
    
    if not swing_highs.empty:
        # Buy Side Liquidity (above swing highs)
        recent_highs = swing_highs.tail(lookback)
        for _, high_point in recent_highs.iterrows():
            age = len(df) - df.index.get_loc(high_point.name)
            if age <= lookback:  # Only recent zones
                liquidity_zones.append({
                    'type': LiquidityType.BSL.value,
                    'price': high_point['high'],
                    'time': high_point['time'],
                    'age': age,
                    'strength': high_point.get('strength', 1.0),
                    'status': 'ACTIVE'
                })
    
    if not swing_lows.empty:
        # Sell Side Liquidity (below swing lows)
        recent_lows = swing_lows.tail(lookback)
        for _, low_point in recent_lows.iterrows():
            age = len(df) - df.index.get_loc(low_point.name)
            if age <= lookback:  # Only recent zones
                liquidity_zones.append({
                    'type': LiquidityType.SSL.value,
                    'price': low_point['low'],
                    'time': low_point['time'],
                    'age': age,
                    'strength': low_point.get('strength', 1.0),
                    'status': 'ACTIVE'
                })
    
    return sorted(liquidity_zones, key=lambda x: x['age'])

def detect_equal_highs_lows(df: pd.DataFrame, tolerance_pips: float = 5.0) -> Dict[str, List[Dict]]:
    """Detect Equal Highs and Equal Lows (EQH/EQL)"""
    if df.empty:
        return {'equal_highs': [], 'equal_lows': []}
    
    swing_highs, swing_lows = identify_swing_points(df)
    
    equal_highs = []
    equal_lows = []
    
    # Convert pips to price difference (assuming 4-digit pricing)
    tolerance = tolerance_pips * 0.0001
    
    # Find Equal Highs
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                high1 = swing_highs.iloc[i]['high']
                high2 = swing_highs.iloc[j]['high']
                if abs(high1 - high2) <= tolerance:
                    equal_highs.append({
                        'type': LiquidityType.EQH.value,
                        'price': (high1 + high2) / 2,
                        'time1': swing_highs.iloc[i]['time'],
                        'time2': swing_highs.iloc[j]['time'],
                        'strength': 2.0  # EQH has higher liquidity
                    })
    
    # Find Equal Lows
    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                low1 = swing_lows.iloc[i]['low']
                low2 = swing_lows.iloc[j]['low']
                if abs(low1 - low2) <= tolerance:
                    equal_lows.append({
                        'type': LiquidityType.EQL.value,
                        'price': (low1 + low2) / 2,
                        'time1': swing_lows.iloc[i]['time'],
                        'time2': swing_lows.iloc[j]['time'],
                        'strength': 2.0
                    })
    
    return {'equal_highs': equal_highs, 'equal_lows': equal_lows}

def detect_liquidity_sweeps(df: pd.DataFrame, liquidity_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect liquidity sweeps: Saat harga menyapu level liquidity."""
    if df.empty:
        return []
    
    sweeps = []
    current_price = df['close'].iloc[-1]
    atr = calculate_atr(df)
    for zone in liquidity_zones:
        if zone['status'] != 'ACTIVE':
            continue
        if zone['type'] == LiquidityType.BSL.value and current_price > zone['price'] + (0.5 * atr):
            sweeps.append({
                'type': LiquidityType.LIQUIDITY_SWEEP.value,
                'direction': 'BULLISH',
                'price': zone['price'],
                'time': df.iloc[-1]['time'],
                'strength': (current_price - zone['price']) / atr
            })
            zone['status'] = 'SWEPT'  # Update status
        elif zone['type'] == LiquidityType.SSL.value and current_price < zone['price'] - (0.5 * atr):
            sweeps.append({
                'type': LiquidityType.LIQUIDITY_SWEEP.value,
                'direction': 'BEARISH',
                'price': zone['price'],
                'time': df.iloc[-1]['time'],
                'strength': (zone['price'] - current_price) / atr
            })
            zone['status'] = 'SWEPT'  # Update status
    return sweeps

# =========================
# Order Block Analysis
# =========================

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> List[Dict[str, Any]]:
    """Detect order blocks: Zona di mana institutional order berkumpul, berdasarkan imbalance candles."""
    if df.empty:
        return []
    
    order_blocks = []
    atr = calculate_atr(df)
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        next_c = df.iloc[i + 1]
        # Bearish OB: Bullish candle diikuti break low (imbalance to downside)
        if curr['close'] > curr['open'] and next_c['low'] < prev['low']:
            strength = (curr['high'] - curr['low']) / atr if atr > 0 else 1.0
            order_blocks.append({
                'type': OrderBlockType.BEARISH_OB.value,
                'high': curr['high'],
                'low': curr['low'],
                'time': curr['time'],
                'strength': strength,
                'status': 'ACTIVE',
                'tested': False
            })
        # Bullish OB: Bearish candle diikuti break high (imbalance to upside)
        elif curr['close'] < curr['open'] and next_c['high'] > prev['high']:
            strength = (curr['high'] - curr['low']) / atr if atr > 0 else 1.0
            order_blocks.append({
                'type': OrderBlockType.BULLISH_OB.value,
                'high': curr['high'],
                'low': curr['low'],
                'time': curr['time'],
                'strength': strength,
                'status': 'ACTIVE',
                'tested': False
            })
    return order_blocks[-lookback:]  # Batasi ke lookback terbaru

def check_order_block_mitigation(df: pd.DataFrame, order_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check if order blocks are mitigated (harga kembali ke OB dan reject)."""
    if df.empty:
        return order_blocks
    
    current_price = df['close'].iloc[-1]
    atr = calculate_atr(df)
    for ob in order_blocks:
        if ob['status'] != 'ACTIVE':
            continue
        ob_mid = (ob['high'] + ob['low']) / 2
        if ob['type'] == OrderBlockType.BULLISH_OB.value:
            if current_price <= ob['high'] and current_price >= ob['low'] - (0.5 * atr):
                ob['tested'] = True
                if current_price > ob_mid:  # Reject to upside
                    ob['status'] = 'MITIGATED'
        elif ob['type'] == OrderBlockType.BEARISH_OB.value:
            if current_price >= ob['low'] and current_price <= ob['high'] + (0.5 * atr):
                ob['tested'] = True
                if current_price < ob_mid:  # Reject to downside
                    ob['status'] = 'MITIGATED'
    return order_blocks

# =========================
# Fair Value Gap Analysis
# =========================

def detect_fair_value_gaps(df: pd.DataFrame, min_gap_atr: float = 0.1) -> List[Dict[str, Any]]:
    """Detect Fair Value Gaps: Gap antara candle yang menunjukkan inefficiency."""
    if df.empty:
        return []
    
    fvgs = []
    atr = calculate_atr(df)
    min_gap = min_gap_atr * atr
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        next_c = df.iloc[i + 1]
        # Bullish FVG: Gap up (low next > high prev)
        if next_c['low'] > prev['high'] + min_gap:
            strength = (next_c['low'] - prev['high']) / atr if atr > 0 else 1.0
            fvgs.append({
                'type': FVGType.BULLISH_FVG.value,
                'top': next_c['low'],
                'bottom': prev['high'],
                'time': curr['time'],
                'strength': strength,
                'status': 'ACTIVE',
                'filled': False
            })
        # Bearish FVG: Gap down (high next < low prev)
        elif next_c['high'] < prev['low'] - min_gap:
            strength = (prev['low'] - next_c['high']) / atr if atr > 0 else 1.0
            fvgs.append({
                'type': FVGType.BEARISH_FVG.value,
                'top': prev['low'],
                'bottom': next_c['high'],
                'time': curr['time'],
                'strength': strength,
                'status': 'ACTIVE',
                'filled': False
            })
    return fvgs

def check_fvg_filling(df: pd.DataFrame, fvgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check if FVGs are filled (harga kembali dan close gap)."""
    if df.empty:
        return fvgs
    
    current_price = df['close'].iloc[-1]
    for fvg in fvgs:
        if fvg['status'] != 'ACTIVE':
            continue
        if fvg['type'] == FVGType.BULLISH_FVG.value:
            if current_price <= fvg['top'] and current_price >= fvg['bottom']:
                fvg['filled'] = True
                fvg['status'] = 'FILLED'
        elif fvg['type'] == FVGType.BEARISH_FVG.value:
            if current_price >= fvg['bottom'] and current_price <= fvg['top']:
                fvg['filled'] = True
                fvg['status'] = 'FILLED'
    return fvgs

# =========================
# Change of Character (CHoCH)
# =========================

def detect_change_of_character(df: pd.DataFrame, structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect Change of Character: Perubahan trend setelah BOS gagal."""
    if df.empty:
        return None
    
    swing_highs, swing_lows = identify_swing_points(df)
    current_price = df['close'].iloc[-1]
    atr = calculate_atr(df)
    
    if structure['trend_direction'] == 'BULLISH' and not swing_lows.empty:
        last_low = swing_lows.iloc[-1]['low']
        prev_low = swing_lows.iloc[-2]['low'] if len(swing_lows) > 1 else last_low
        if current_price < last_low and last_low < prev_low:
            return {
                'type': MarketStructure.CHOCH_BEARISH.value,
                'price': last_low,
                'time': df.iloc[-1]['time'],
                'strength': (last_low - current_price) / atr if atr > 0 else 1.0,
                'previous_trend': 'BULLISH',
                'new_trend': 'BEARISH'
            }
    elif structure['trend_direction'] == 'BEARISH' and not swing_highs.empty:
        last_high = swing_highs.iloc[-1]['high']
        prev_high = swing_highs.iloc[-2]['high'] if len(swing_highs) > 1 else last_high
        if current_price > last_high and last_high > prev_high:
            return {
                'type': MarketStructure.CHOCH_BULLISH.value,
                'price': last_high,
                'time': df.iloc[-1]['time'],
                'strength': (current_price - last_high) / atr if atr > 0 else 1.0,
                'previous_trend': 'BEARISH',
                'new_trend': 'BULLISH'
            }
    return None

# =========================
# Premium/Discount Zones
# =========================

def calculate_premium_discount_zones(df: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
    """Calculate Premium and Discount zones based on recent price action"""
    if df.empty:
        return {'premium': {}, 'equilibrium': {}, 'discount': {}, 'current_zone': 'NEUTRAL', 'range_high': None, 'range_low': None}
    
    if len(df) < lookback:
        lookback = len(df)
    
    recent_data = df.tail(lookback)
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    range_size = high - low
    
    # Standard SMC levels
    premium_zone = {
        'top': high,
        'bottom': high - (range_size * 0.2),  # Top 20%
        'mid': high - (range_size * 0.1)  # Top 10%
    }
    
    equilibrium_zone = {
        'top': low + (range_size * 0.6),
        'bottom': low + (range_size * 0.4),
        'mid': low + (range_size * 0.5)  # 50% level
    }
    
    discount_zone = {
        'top': low + (range_size * 0.2),  # Bottom 20%
        'bottom': low,
        'mid': low + (range_size * 0.1)  # Bottom 10%
    }
    
    current_price = df['close'].iloc[-1]
    
    # Determine current zone
    if current_price >= premium_zone['bottom']:
        current_zone = 'PREMIUM'
    elif current_price <= discount_zone['top']:
        current_zone = 'DISCOUNT'
    else:
        current_zone = 'EQUILIBRIUM'
    
    return {
        'premium': premium_zone,
        'equilibrium': equilibrium_zone,
        'discount': discount_zone,
        'current_zone': current_zone,
        'range_high': high,
        'range_low': low
    }

# =========================
# Main SMC Analysis Function
# =========================

def analyze_smc_full(df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Complete SMC analysis combining all elements"""
    if df.empty:
        return {}
    
    if config is None:
        config = {
            'swing_length': 5,
            'liquidity_lookback': 20,
            'order_block_lookback': 50,
            'min_gap_atr': 0.1,
            'eq_tolerance_pips': 5.0
        }
    
    # Core analysis
    structure = analyze_market_structure(df, config['swing_length'])
    liquidity_zones = detect_liquidity_zones(df, config['liquidity_lookback'])
    equal_levels = detect_equal_highs_lows(df, config['eq_tolerance_pips'])
    liquidity_sweeps = detect_liquidity_sweeps(df, liquidity_zones)
    
    order_blocks = detect_order_blocks(df, config['order_block_lookback'])
    order_blocks = check_order_block_mitigation(df, order_blocks)
    
    fvgs = detect_fair_value_gaps(df, config['min_gap_atr'])
    fvgs = check_fvg_filling(df, fvgs)
    
    choch = detect_change_of_character(df, structure)
    zones = calculate_premium_discount_zones(df)
    
    # Calculate overall bias
    bullish_signals = 0
    bearish_signals = 0
    
    # Structure bias
    if structure['trend_direction'] == 'BULLISH':
        bullish_signals += 3
    elif structure['trend_direction'] == 'BEARISH':
        bearish_signals += 3
    
    # Liquidity sweep bias
    for sweep in liquidity_sweeps:
        if sweep['direction'] == 'BULLISH':
            bullish_signals += 2
        elif sweep['direction'] == 'BEARISH':
            bearish_signals += 2
    
    # CHoCH bias
    if choch:
        if 'BULLISH' in choch['type']:
            bullish_signals += 2
        elif 'BEARISH' in choch['type']:
            bearish_signals += 2
    
    # Zone bias
    if zones['current_zone'] == 'DISCOUNT':
        bullish_signals += 1  # Discount = potential buy zone
    elif zones['current_zone'] == 'PREMIUM':
        bearish_signals += 1  # Premium = potential sell zone
    
    # Determine overall bias
    total_signals = bullish_signals + bearish_signals
    if total_signals > 0:
        bullish_percentage = bullish_signals / total_signals
        if bullish_percentage >= 0.7:
            overall_bias = 'STRONG_BULLISH'
        elif bullish_percentage >= 0.6:
            overall_bias = 'BULLISH'
        elif bullish_percentage <= 0.3:
            overall_bias = 'STRONG_BEARISH'
        elif bullish_percentage <= 0.4:
            overall_bias = 'BEARISH'
        else:
            overall_bias = 'NEUTRAL'
    else:
        overall_bias = 'NEUTRAL'
    
    return {
        'timestamp': df.iloc[-1]['time'] if 'time' in df.columns else datetime.now(),
        'current_price': df['close'].iloc[-1],
        'atr': calculate_atr(df),
        'overall_bias': overall_bias,
        'bias_strength': max(bullish_signals, bearish_signals),
        'structure': structure,
        'liquidity_zones': liquidity_zones,
        'equal_levels': equal_levels,
        'liquidity_sweeps': liquidity_sweeps,
        'order_blocks': [ob for ob in order_blocks if ob['status'] == 'ACTIVE'],
        'fair_value_gaps': [fvg for fvg in fvgs if fvg['status'] == 'ACTIVE'],
        'change_of_character': choch,
        'premium_discount_zones': zones,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals
    }

# =========================
# Entry Point Calculation
# =========================

def calculate_optimal_entry_exit(smc_analysis: Dict[str, Any], risk_reward_ratio: float = 2.0) -> Dict[str, Any]:
    """Calculate optimal entry and exit points based on SMC analysis"""
    if not smc_analysis:
        return {'direction': 'WAIT', 'entry_zones': []}
    
    current_price = smc_analysis['current_price']
    atr = smc_analysis['atr']
    bias = smc_analysis['overall_bias']
    
    if 'BULLISH' in bias:
        # Look for bullish entry opportunities
        entry_zones = []
        
        # Order blocks as entry zones
        for ob in smc_analysis['order_blocks']:
            if ob['type'] == OrderBlockType.BULLISH_OB.value and not ob['tested']:
                entry_price = (ob['high'] + ob['low']) / 2
                stop_loss = ob['low'] - (atr * 0.5)
                take_profit = entry_price + (abs(entry_price - stop_loss) * risk_reward_ratio)
                entry_zones.append({
                    'type': 'ORDER_BLOCK',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': ob['strength']
                })
        
        # Fair Value Gaps as entry zones
        for fvg in smc_analysis['fair_value_gaps']:
            if fvg['type'] == FVGType.BULLISH_FVG.value and not fvg['filled']:
                entry_price = (fvg['top'] + fvg['bottom']) / 2
                stop_loss = fvg['bottom'] - (atr * 0.5)
                take_profit = entry_price + (abs(entry_price - stop_loss) * risk_reward_ratio)
                entry_zones.append({
                    'type': 'FAIR_VALUE_GAP',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': fvg['strength']
                })
        
        # Discount zones as entry areas
        if smc_analysis['premium_discount_zones']['current_zone'] == 'DISCOUNT':
            discount_zone = smc_analysis['premium_discount_zones']['discount']
            entry_price = discount_zone['mid']
            stop_loss = discount_zone['bottom'] - (atr * 0.5)
            take_profit = entry_price + (abs(entry_price - stop_loss) * risk_reward_ratio)
            entry_zones.append({
                'type': 'DISCOUNT_ZONE',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 1.5
            })
        
        return {
            'direction': 'BUY',
            'entry_zones': sorted(entry_zones, key=lambda x: x['confidence'], reverse=True)
        }
    
    elif 'BEARISH' in bias:
        # Look for bearish entry opportunities
        entry_zones = []
        
        # Order blocks as entry zones
        for ob in smc_analysis['order_blocks']:
            if ob['type'] == OrderBlockType.BEARISH_OB.value and not ob['tested']:
                entry_price = (ob['high'] + ob['low']) / 2
                stop_loss = ob['high'] + (atr * 0.5)
                take_profit = entry_price - (abs(stop_loss - entry_price) * risk_reward_ratio)
                entry_zones.append({
                    'type': 'ORDER_BLOCK',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': ob['strength']
                })
        
        # Fair Value Gaps as entry zones
        for fvg in smc_analysis['fair_value_gaps']:
            if fvg['type'] == FVGType.BEARISH_FVG.value and not fvg['filled']:
                entry_price = (fvg['top'] + fvg['bottom']) / 2
                stop_loss = fvg['top'] + (atr * 0.5)
                take_profit = entry_price - (abs(stop_loss - entry_price) * risk_reward_ratio)
                entry_zones.append({
                    'type': 'FAIR_VALUE_GAP',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': fvg['strength']
                })
        
        # Premium zones as entry areas
        if smc_analysis['premium_discount_zones']['current_zone'] == 'PREMIUM':
            premium_zone = smc_analysis['premium_discount_zones']['premium']
            entry_price = premium_zone['mid']
            stop_loss = premium_zone['top'] + (atr * 0.5)
            take_profit = entry_price - (abs(stop_loss - entry_price) * risk_reward_ratio)
            entry_zones.append({
                'type': 'PREMIUM_ZONE',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 1.5
            })
        
        return {
            'direction': 'SELL',
            'entry_zones': sorted(entry_zones, key=lambda x: x['confidence'], reverse=True)
        }
    
    return {
        'direction': 'WAIT',
        'entry_zones': []
    }