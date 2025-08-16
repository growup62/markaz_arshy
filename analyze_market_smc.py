# analyze_market_smc.py
# Enhanced Market Analysis using Pure Smart Money Concepts (SMC)
# No Traditional Indicators - Focus on Market Structure, Liquidity, and Institutional Behavior

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf

# Import our Pure SMC Technical Indicators
from technical_indicators_smc import (
    analyze_smc_full,
    calculate_optimal_entry_exit,
    MarketStructure,
    LiquidityType,
    OrderBlockType,
    FVGType
)

# =========================
# Trading Profile Enums
# =========================

class TradingProfile(Enum):
    SCALPER = "scalper"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"

class MarketPhase(Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    TRANSITION = "transition"

# =========================
# Pair-Specific Configurations
# =========================

SPREAD_RULES = {
    "XAUUSD": {"max_abs": 0.5, "max_rel_atr": 0.25, "smc_sensitivity": 1.2},
    "BTCUSD": {"max_abs": 30.0, "max_rel_atr": 0.12, "smc_sensitivity": 1.0},
    "EURUSD": {"max_abs": 0.00015, "max_rel_atr": 0.20, "smc_sensitivity": 0.8},
    "GBPUSD": {"max_abs": 0.00020, "max_rel_atr": 0.25, "smc_sensitivity": 0.9},
    "USDJPY": {"max_abs": 0.015, "max_rel_atr": 0.18, "smc_sensitivity": 0.85},
    "DEFAULT": {"max_rel_atr": 0.20, "smc_sensitivity": 1.0}
}

# SMC-specific configurations for different pairs
SMC_CONFIGS = {
    "XAUUSD": {
        "swing_length": 7,  # Gold needs longer swing detection
        "liquidity_lookback": 30,
        "order_block_lookback": 100,
        "min_gap_atr": 0.15,
        "eq_tolerance_pips": 10.0
    },
    "BTCUSD": {
        "swing_length": 5,
        "liquidity_lookback": 20,
        "order_block_lookback": 50,
        "min_gap_atr": 0.08,
        "eq_tolerance_pips": 50.0  # Crypto needs higher tolerance
    },
    "EURUSD": {
        "swing_length": 4,
        "liquidity_lookback": 15,
        "order_block_lookback": 40,
        "min_gap_atr": 0.12,
        "eq_tolerance_pips": 3.0
    },
    "DEFAULT": {
        "swing_length": 5,
        "liquidity_lookback": 20,
        "order_block_lookback": 50,
        "min_gap_atr": 0.1,
        "eq_tolerance_pips": 5.0
    }
}

# =========================
# Utility Functions
# =========================

def _match_symbol_key(symbol: str) -> str:
    """Match symbol to configuration key"""
    if not symbol:
        return "DEFAULT"
    s = str(symbol).upper()
    for k in SPREAD_RULES.keys():
        if k != "DEFAULT" and s.startswith(k):
            return k
    return "DEFAULT"

def is_spread_ok_for_symbol(symbol: str, spread: float, atr: float) -> bool:
    """SMC-aware spread validation"""
    try:
        if spread is None or (isinstance(spread, float) and spread != spread):
            return False
        
        key = _match_symbol_key(symbol)
        rules = SPREAD_RULES.get(key, SPREAD_RULES["DEFAULT"])
        
        # SMC sensitivity adjustment
        smc_sensitivity = rules.get("smc_sensitivity", 1.0)
        
        if atr is not None and atr > 0:
            rel = float(spread) / float(atr)
            max_rel = rules.get("max_rel_atr", 0.20) * smc_sensitivity
            if rel > max_rel:
                return False
        
        max_abs = rules.get("max_abs")
        if max_abs is not None:
            adjusted_max = max_abs * smc_sensitivity
            if float(spread) > adjusted_max:
                return False
        
        return True
    except Exception:
        return False

def fetch_data(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """Enhanced data fetching with SMC requirements"""
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data available for ticker '{ticker}'")
        
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df.index.name = 'time'
        df = df.reset_index()
        
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])
        
        # Ensure time column exists for SMC analysis  
        if 'time' not in df.columns:
            df['time'] = df.index
        
        # Convert time to datetime if it's not already
        if 'time' in df.columns:
            try:
                df['time'] = pd.to_datetime(df['time'])
            except:
                df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
        
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# =========================
# Enhanced Profile Analyzer
# =========================

class SMCProfileAnalyzer:
    """SMC-based market analysis for different trading profiles"""
    
    def __init__(self, profile: TradingProfile, symbol: str = "DEFAULT"):
        self.profile = profile
        self.symbol = symbol
        self.smc_config = SMC_CONFIGS.get(_match_symbol_key(symbol), SMC_CONFIGS["DEFAULT"])
        
        # Profile-specific SMC adjustments
        if profile == TradingProfile.SCALPER:
            self.smc_config = self.smc_config.copy()
            self.smc_config['swing_length'] = max(3, self.smc_config['swing_length'] - 2)
            self.smc_config['liquidity_lookback'] = min(10, self.smc_config['liquidity_lookback'])
            self.smc_config['min_gap_atr'] = self.smc_config['min_gap_atr'] * 0.8
        
        elif profile == TradingProfile.POSITION:
            self.smc_config = self.smc_config.copy()
            self.smc_config['swing_length'] = self.smc_config['swing_length'] + 3
            self.smc_config['liquidity_lookback'] = self.smc_config['liquidity_lookback'] * 2
            self.smc_config['order_block_lookback'] = self.smc_config['order_block_lookback'] * 2
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete SMC analysis based on trading profile"""
        if len(df) < 50:  # Minimum data requirement
            return self._empty_analysis()
        
        # Core SMC Analysis
        smc_analysis = analyze_smc_full(df, self.smc_config)
        
        # Profile-specific interpretation
        if self.profile == TradingProfile.SCALPER:
            return self._analyze_scalper(df, smc_analysis)
        elif self.profile == TradingProfile.INTRADAY:
            return self._analyze_intraday(df, smc_analysis)
        elif self.profile == TradingProfile.SWING:
            return self._analyze_swing(df, smc_analysis)
        else:  # POSITION
            return self._analyze_position(df, smc_analysis)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when insufficient data"""
        return {
            'smc_bias': 'NEUTRAL',
            'confidence': 0.0,
            'entry_signals': [],
            'risk_level': 'HIGH',
            'market_phase': MarketPhase.TRANSITION.value,
            'key_levels': {},
            'warnings': ['Insufficient data for SMC analysis']
        }
    
    def _analyze_scalper(self, df: pd.DataFrame, smc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Scalper-focused SMC analysis"""
        result = {
            'smc_bias': smc_analysis['overall_bias'],
            'confidence': min(smc_analysis['bias_strength'] / 5.0, 1.0),
            'entry_signals': [],
            'risk_level': 'MEDIUM',
            'market_phase': self._determine_scalper_phase(smc_analysis),
            'warnings': []
        }
        
        # Focus on recent FVGs and Order Blocks for scalping
        recent_fvgs = [fvg for fvg in smc_analysis['fair_value_gaps'] if fvg['strength'] > 1.0]
        recent_obs = [ob for ob in smc_analysis['order_blocks'] if not ob['tested']]
        
        if recent_fvgs:
            result['entry_signals'].append({
                'type': 'FVG_RETEST',
                'zones': recent_fvgs[:2],  # Top 2 FVGs
                'timeframe': 'M1-M5'
            })
        
        if recent_obs:
            result['entry_signals'].append({
                'type': 'ORDER_BLOCK_RETEST',
                'zones': recent_obs[:1],  # Top 1 OB
                'timeframe': 'M1-M15'
            })
        
        # Scalper warnings
        if smc_analysis['premium_discount_zones']['current_zone'] == 'EQUILIBRIUM':
            result['warnings'].append('Price in equilibrium - consider waiting for clear bias')
        
        return result
    
    def _analyze_intraday(self, df: pd.DataFrame, smc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Intraday trader SMC analysis"""
        result = {
            'smc_bias': smc_analysis['overall_bias'],
            'confidence': min(smc_analysis['bias_strength'] / 7.0, 1.0),
            'entry_signals': [],
            'risk_level': 'MEDIUM',
            'market_phase': self._determine_intraday_phase(smc_analysis),
            'session_analysis': self._analyze_trading_session(df),
            'warnings': []
        }
        
        # Liquidity sweep opportunities
        if smc_analysis['liquidity_sweeps']:
            strong_sweeps = [s for s in smc_analysis['liquidity_sweeps'] if s['strength'] > 1.5]
            if strong_sweeps:
                result['entry_signals'].append({
                    'type': 'LIQUIDITY_SWEEP_REVERSAL',
                    'sweeps': strong_sweeps,
                    'timeframe': 'M15-H1'
                })
        
        # Premium/Discount zone analysis
        zones = smc_analysis['premium_discount_zones']
        if zones['current_zone'] in ['PREMIUM', 'DISCOUNT']:
            result['entry_signals'].append({
                'type': 'PREMIUM_DISCOUNT_REVERSAL',
                'zone': zones['current_zone'],
                'levels': zones[zones['current_zone'].lower()],
                'timeframe': 'H1-H4'
            })
        
        return result
    
    def _analyze_swing(self, df: pd.DataFrame, smc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Swing trader SMC analysis"""
        result = {
            'smc_bias': smc_analysis['overall_bias'],
            'confidence': min(smc_analysis['bias_strength'] / 8.0, 1.0),
            'entry_signals': [],
            'risk_level': 'LOW',
            'market_phase': self._determine_swing_phase(smc_analysis),
            'structure_analysis': smc_analysis['structure'],
            'warnings': []
        }
        
        # Structure-based signals
        structure = smc_analysis['structure']
        if structure['bos_detected']:
            result['entry_signals'].append({
                'type': 'BREAK_OF_STRUCTURE',
                'direction': structure['trend_direction'],
                'strength': structure['structure_strength'],
                'timeframe': 'H4-D1'
            })
        
        # CHoCH signals
        if smc_analysis['change_of_character']:
            choch = smc_analysis['change_of_character']
            result['entry_signals'].append({
                'type': 'CHANGE_OF_CHARACTER',
                'direction': choch['new_trend'],
                'strength': choch['strength'],
                'timeframe': 'H4-D1'
            })
        
        # High-quality Order Blocks
        strong_obs = [ob for ob in smc_analysis['order_blocks'] 
                     if ob['strength'] > 2.0 and not ob['tested']]
        if strong_obs:
            result['entry_signals'].append({
                'type': 'HIGH_QUALITY_ORDER_BLOCK',
                'blocks': strong_obs[:2],
                'timeframe': 'H1-H4'
            })
        
        return result
    
    def _analyze_position(self, df: pd.DataFrame, smc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Position trader SMC analysis"""
        result = {
            'smc_bias': smc_analysis['overall_bias'],
            'confidence': min(smc_analysis['bias_strength'] / 10.0, 1.0),
            'entry_signals': [],
            'risk_level': 'LOW',
            'market_phase': self._determine_position_phase(smc_analysis),
            'long_term_structure': self._analyze_long_term_structure(df, smc_analysis),
            'warnings': []
        }
        
        # Major structure changes
        if smc_analysis['structure']['bos_detected'] and smc_analysis['structure']['structure_strength'] > 0.7:
            result['entry_signals'].append({
                'type': 'MAJOR_STRUCTURE_SHIFT',
                'direction': smc_analysis['structure']['trend_direction'],
                'strength': smc_analysis['structure']['structure_strength'],
                'timeframe': 'D1-W1'
            })
        
        # Institutional Order Blocks
        institutional_obs = [ob for ob in smc_analysis['order_blocks'] 
                           if ob['strength'] > 3.0]
        if institutional_obs:
            result['entry_signals'].append({
                'type': 'INSTITUTIONAL_ORDER_BLOCK',
                'blocks': institutional_obs,
                'timeframe': 'H4-D1'
            })
        
        return result
    
    def _analyze_trading_session(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trading session impact"""
        if 'time' not in df.columns:
            return {'session': 'UNKNOWN', 'activity': 'NORMAL'}
        
        current_time = pd.to_datetime(df['time'].iloc[-1])
        utc_hour = current_time.hour
        
        if 0 <= utc_hour < 8:
            session = "ASIAN"
            activity = "LOW" if utc_hour < 2 else "MEDIUM"
        elif 8 <= utc_hour < 16:
            session = "LONDON"
            activity = "HIGH"
        else:
            session = "NEW_YORK"
            activity = "HIGH" if utc_hour < 20 else "MEDIUM"
        
        return {'session': session, 'activity': activity}
    
    def _determine_scalper_phase(self, smc_analysis: Dict[str, Any]) -> str:
        """Determine market phase for scalpers"""
        if smc_analysis['liquidity_sweeps']:
            return MarketPhase.TRANSITION.value
        elif smc_analysis['premium_discount_zones']['current_zone'] == 'EQUILIBRIUM':
            return MarketPhase.ACCUMULATION.value
        else:
            return MarketPhase.MARKUP.value if 'BULLISH' in smc_analysis['overall_bias'] else MarketPhase.MARKDOWN.value
    
    def _determine_intraday_phase(self, smc_analysis: Dict[str, Any]) -> str:
        """Determine market phase for intraday traders"""
        structure_strength = smc_analysis['structure']['structure_strength']
        
        if structure_strength > 0.7:
            return MarketPhase.MARKUP.value if 'BULLISH' in smc_analysis['overall_bias'] else MarketPhase.MARKDOWN.value
        elif structure_strength > 0.4:
            return MarketPhase.DISTRIBUTION.value if smc_analysis['premium_discount_zones']['current_zone'] == 'PREMIUM' else MarketPhase.ACCUMULATION.value
        else:
            return MarketPhase.TRANSITION.value
    
    def _determine_swing_phase(self, smc_analysis: Dict[str, Any]) -> str:
        """Determine market phase for swing traders"""
        if smc_analysis['change_of_character']:
            return MarketPhase.TRANSITION.value
        elif smc_analysis['structure']['bos_detected']:
            return MarketPhase.MARKUP.value if 'BULLISH' in smc_analysis['overall_bias'] else MarketPhase.MARKDOWN.value
        else:
            zone = smc_analysis['premium_discount_zones']['current_zone']
            if zone == 'PREMIUM':
                return MarketPhase.DISTRIBUTION.value
            elif zone == 'DISCOUNT':
                return MarketPhase.ACCUMULATION.value
            else:
                return MarketPhase.TRANSITION.value
    
    def _determine_position_phase(self, smc_analysis: Dict[str, Any]) -> str:
        """Determine market phase for position traders"""
        bias_strength = smc_analysis['bias_strength']
        
        if bias_strength >= 8:
            return MarketPhase.MARKUP.value if 'BULLISH' in smc_analysis['overall_bias'] else MarketPhase.MARKDOWN.value
        elif bias_strength >= 5:
            zone = smc_analysis['premium_discount_zones']['current_zone']
            if zone == 'PREMIUM' and 'BEARISH' in smc_analysis['overall_bias']:
                return MarketPhase.DISTRIBUTION.value
            elif zone == 'DISCOUNT' and 'BULLISH' in smc_analysis['overall_bias']:
                return MarketPhase.ACCUMULATION.value
            else:
                return MarketPhase.TRANSITION.value
        else:
            return MarketPhase.ACCUMULATION.value
    
    def _analyze_long_term_structure(self, df: pd.DataFrame, smc_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze long-term market structure for position trading"""
        # Use longer lookback for position trading
        long_term_high = df['high'].tail(200).max() if len(df) >= 200 else df['high'].max()
        long_term_low = df['low'].tail(200).min() if len(df) >= 200 else df['low'].min()
        current_price = df['close'].iloc[-1]
        
        # Position in long-term range
        range_size = long_term_high - long_term_low
        position_pct = (current_price - long_term_low) / range_size if range_size > 0 else 0.5
        
        return {
            'long_term_high': long_term_high,
            'long_term_low': long_term_low,
            'range_position_pct': position_pct,
            'in_upper_third': position_pct > 0.67,
            'in_lower_third': position_pct < 0.33,
            'trend_bias': 'BULLISH' if position_pct > 0.6 else 'BEARISH' if position_pct < 0.4 else 'NEUTRAL'
        }

# =========================
# Enhanced Market Context
# =========================

def get_market_context_smc(df: pd.DataFrame, profile: Optional[str] = None, symbol: str = "DEFAULT") -> Dict[str, Any]:
    """Enhanced market context analysis using pure SMC"""
    if df.empty or len(df) < 50:
        return {
            'error': 'Insufficient data for SMC analysis',
            'min_required': 50,
            'current_length': len(df)
        }
    
    # Get SMC configuration for symbol
    smc_config = SMC_CONFIGS.get(_match_symbol_key(symbol), SMC_CONFIGS["DEFAULT"])
    
    # Core SMC Analysis
    smc_analysis = analyze_smc_full(df, smc_config)
    
    # Calculate optimal entry/exit points
    entry_exit = calculate_optimal_entry_exit(smc_analysis, risk_reward_ratio=2.0)
    
    # Profile-specific analysis
    profile_analysis = None
    if profile:
        try:
            profile_enum = TradingProfile(profile.lower())
            analyzer = SMCProfileAnalyzer(profile_enum, symbol)
            profile_analysis = analyzer.analyze(df)
        except ValueError:
            logging.warning(f"Unknown trading profile: {profile}")
    
    # Enhanced context
    context = {
        'timestamp': datetime.now(),
        'symbol': symbol,
        'current_price': smc_analysis['current_price'],
        'atr': smc_analysis['atr'],
        
        # Core SMC Analysis
        'smc_analysis': smc_analysis,
        'entry_exit_analysis': entry_exit,
        
        # Market bias with SMC reasoning
        'market_bias': smc_analysis['overall_bias'],
        'bias_strength': smc_analysis['bias_strength'],
        'bias_components': {
            'bullish_signals': smc_analysis['bullish_signals'],
            'bearish_signals': smc_analysis['bearish_signals']
        },
        
        # Key SMC levels
        'key_levels': {
            'premium_discount': smc_analysis['premium_discount_zones'],
            'active_order_blocks': len([ob for ob in smc_analysis['order_blocks'] if ob['status'] == 'ACTIVE']),
            'active_fvgs': len([fvg for fvg in smc_analysis['fair_value_gaps'] if fvg['status'] == 'ACTIVE']),
            'liquidity_zones': len(smc_analysis['liquidity_zones'])
        },
        
        # Profile-specific insights
        'profile_analysis': profile_analysis,
        
        # Risk assessment
        'risk_assessment': {
            'overall_risk': _calculate_overall_risk(smc_analysis),
            'volatility_risk': 'HIGH' if smc_analysis['atr'] > df['close'].iloc[-1] * 0.02 else 'NORMAL',
            'structure_risk': 'LOW' if smc_analysis['structure']['structure_strength'] > 0.6 else 'MEDIUM'
        }
    }
    
    return context

def _calculate_overall_risk(smc_analysis: Dict[str, Any]) -> str:
    """Calculate overall risk based on SMC factors"""
    risk_factors = 0
    
    # High bias strength = lower risk
    if smc_analysis['bias_strength'] < 3:
        risk_factors += 1
    
    # Conflicting signals = higher risk
    total_signals = smc_analysis['bullish_signals'] + smc_analysis['bearish_signals']
    if total_signals > 0:
        balance = abs(smc_analysis['bullish_signals'] - smc_analysis['bearish_signals']) / total_signals
        if balance < 0.4:  # Signals are too balanced
            risk_factors += 1
    
    # Equilibrium zone = higher risk
    if smc_analysis['premium_discount_zones']['current_zone'] == 'EQUILIBRIUM':
        risk_factors += 1
    
    # No clear structure = higher risk
    if smc_analysis['structure']['structure_strength'] < 0.3:
        risk_factors += 1
    
    if risk_factors >= 3:
        return 'HIGH'
    elif risk_factors >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'

# =========================
# Backtesting with SMC
# =========================

def backtest_smc_strategy(df: pd.DataFrame, context: Dict[str, Any], 
                         initial_capital: float = 10000, 
                         risk_per_trade: float = 0.01,
                         symbol: str = "DEFAULT") -> Dict[str, Any]:
    """Enhanced backtesting using SMC analysis"""
    trades = []
    equity = [initial_capital]
    positions = []
    
    # Get symbol-specific spread rules
    spread_rules = SPREAD_RULES.get(_match_symbol_key(symbol), SPREAD_RULES["DEFAULT"])
    
    for i in range(100, len(df)):  # Start after sufficient data
        current_data = df.iloc[:i+1]
        
        # Get SMC analysis for current point
        sub_context = get_market_context_smc(current_data, symbol=symbol)
        
        if 'error' in sub_context:
            continue
        
        smc_analysis = sub_context['smc_analysis']
        entry_exit = sub_context['entry_exit_analysis']
        
        current_price = df['close'].iloc[i]
        atr = smc_analysis['atr']
        
        # Check spread conditions
        spread_val = df.get('spread', pd.Series([0.001] * len(df))).iloc[i]
        if not is_spread_ok_for_symbol(symbol, spread_val, atr):
            continue
        
        # Entry logic based on SMC
        if not positions and entry_exit['direction'] != 'WAIT':
            if len(entry_exit['entry_zones']) > 0:
                best_zone = entry_exit['entry_zones'][0]
                
                # Check if current price is near entry zone
                entry_price = best_zone['entry_price']
                if abs(current_price - entry_price) / current_price < 0.01:  # Within 1%
                    
                    # Calculate position size based on risk
                    stop_loss = best_zone.get('stop_loss', entry_price - atr * 2)
                    risk_amount = initial_capital * risk_per_trade
                    
                    if entry_exit['direction'] == 'BUY':
                        risk_per_share = entry_price - stop_loss
                    else:
                        risk_per_share = stop_loss - entry_price
                    
                    if risk_per_share > 0:
                        position_size = risk_amount / risk_per_share
                        
                        positions.append({
                            'direction': entry_exit['direction'],
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': best_zone.get('take_profit'),
                            'size': position_size,
                            'entry_time': df['time'].iloc[i] if 'time' in df.columns else i,
                            'smc_reason': best_zone['type']
                        })
        
        # Exit logic
        elif positions:
            pos = positions[0]
            pnl = 0
            exit_reason = None
            
            if pos['direction'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    pnl = pos['size'] * (current_price - pos['entry_price'])
                    exit_reason = 'STOP_LOSS'
                elif pos['take_profit'] and current_price >= pos['take_profit']:
                    pnl = pos['size'] * (current_price - pos['entry_price'])
                    exit_reason = 'TAKE_PROFIT'
            else:  # SELL
                if current_price >= pos['stop_loss']:
                    pnl = pos['size'] * (pos['entry_price'] - current_price)
                    exit_reason = 'STOP_LOSS'
                elif pos['take_profit'] and current_price <= pos['take_profit']:
                    pnl = pos['size'] * (pos['entry_price'] - current_price)
                    exit_reason = 'TAKE_PROFIT'
            
            if exit_reason:
                trades.append({
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'direction': pos['direction'],
                    'exit_reason': exit_reason,
                    'smc_reason': pos['smc_reason'],
                    'hold_time': i - df[df['time'] == pos['entry_time']].index[0] if 'time' in df.columns else 1
                })
                positions = []
        
        # Update equity
        current_equity = equity[-1] + (trades[-1]['pnl'] if trades else 0)
        equity.append(current_equity)
    
    # Calculate metrics
    if not trades:
        return {'error': 'No trades generated', 'trades': 0}
    
    total_pnl = sum(t['pnl'] for t in trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades)
    
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    return {
        'total_pnl': total_pnl,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
        'final_equity': equity[-1],
        'max_drawdown': min(equity) - max(equity[:equity.index(min(equity))+1]) if len(equity) > 1 else 0,
        'smc_strategy_breakdown': {
            'order_block_trades': len([t for t in trades if 'ORDER_BLOCK' in t['smc_reason']]),
            'fvg_trades': len([t for t in trades if 'FVG' in t['smc_reason']]),
            'zone_trades': len([t for t in trades if 'ZONE' in t['smc_reason']])
        }
    }

# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Test with Bitcoin data
    print("Testing SMC Analysis with BTC-USD...")
    df_btc = fetch_data('BTC-USD', period='3mo', interval='1h')
    
    if not df_btc.empty:
        context_btc = get_market_context_smc(df_btc, profile='swing', symbol='BTCUSD')
        print(f"\nBTC-USD SMC Analysis:")
        print(f"Market Bias: {context_btc['market_bias']}")
        print(f"Bias Strength: {context_btc['bias_strength']}")
        print(f"Current Zone: {context_btc['smc_analysis']['premium_discount_zones']['current_zone']}")
        print(f"Active Order Blocks: {context_btc['key_levels']['active_order_blocks']}")
        print(f"Active FVGs: {context_btc['key_levels']['active_fvgs']}")
        
        if context_btc['entry_exit_analysis']['direction'] != 'WAIT':
            direction = context_btc['entry_exit_analysis']['direction']
            zones = context_btc['entry_exit_analysis']['entry_zones']
            print(f"\nEntry Recommendation: {direction}")
            if zones:
                best_zone = zones[0]
                print(f"Best Entry Zone: {best_zone['type']} at {best_zone['entry_price']:.2f}")
        
        # Run backtest
        backtest_results = backtest_smc_strategy(df_btc, context_btc, symbol='BTCUSD')
        if 'error' not in backtest_results:
            print(f"\nBacktest Results:")
            print(f"Total PnL: ${backtest_results['total_pnl']:.2f}")
            print(f"Win Rate: {backtest_results['win_rate']:.1%}")
            print(f"Total Trades: {backtest_results['total_trades']}")
    
    # Test with Gold data
    print("\n" + "="*50)
    print("Testing SMC Analysis with Gold (GC=F)...")
    df_gold = fetch_data('GC=F', period='3mo', interval='1h')
    
    if not df_gold.empty:
        context_gold = get_market_context_smc(df_gold, profile='intraday', symbol='XAUUSD')
        print(f"\nXAU-USD SMC Analysis:")
        print(f"Market Bias: {context_gold['market_bias']}")
        print(f"Risk Level: {context_gold['risk_assessment']['overall_risk']}")
        
        if context_gold.get('profile_analysis'):
            profile_data = context_gold['profile_analysis']
            print(f"Market Phase: {profile_data['market_phase']}")
            print(f"Entry Signals: {len(profile_data['entry_signals'])}")
