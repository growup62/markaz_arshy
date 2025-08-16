# smc_mt5_integration.py
# Smart Money Concepts Integration with MetaTrader 5
# Menghubungkan sistem SMC dengan MT5 dan signal broadcasting

import MetaTrader5 as mt5
import json
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import requests

# Import SMC modules
from analyze_market_smc import get_market_context_smc, SMC_CONFIGS, _match_symbol_key
from technical_indicators_smc import analyze_smc_full

class SMC_MT5_Integration:
    """Integrasi sistem SMC dengan MetaTrader 5 untuk trading otomatis"""
    
    def __init__(self, mt5_path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"):
        self.mt5_path = mt5_path
        self.last_signals = {}
        self.active_trades = {}
        self.smc_history = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SMC_MT5')
        
        # MT5 symbol mappings
        self.symbol_mappings = {
            'BTCUSD': 'BTCUSD',
            'XAUUSD': 'XAUUSD',
            'EURUSD': 'EURUSD',
            'GBPUSD': 'GBPUSD',
            'USDJPY': 'USDJPY',
            'AUDUSD': 'AUDUSD',
            'USDCAD': 'USDCAD',
            'USDCHF': 'USDCHF'
        }
        
        # Risk management settings
        self.risk_settings = {
            'max_risk_per_trade': 0.02,  # 2% per trade
            'max_positions': 5,
            'max_daily_trades': 10,
            'stop_loss_multiplier': 2.0,
            'take_profit_multiplier': 3.0
        }
        
    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        if not mt5.initialize(path=self.mt5_path):
            self.logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            return False
        
        self.logger.info("✅ MT5 initialized successfully")
        return True
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        self.logger.info("🔐 MT5 connection closed")
    
    def get_mt5_data(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Get OHLC data from MT5"""
        if not self.initialize_mt5():
            return pd.DataFrame()
        
        try:
            # Map timeframe
            tf_mapping = {
                '1m': mt5.TIMEFRAME_M1,
                '5m': mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30,
                '1h': mt5.TIMEFRAME_H1,
                '4h': mt5.TIMEFRAME_H4,
                '1d': mt5.TIMEFRAME_D1
            }
            
            mt5_tf = tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get data
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'tick_volume': 'volume'})
            
            self.logger.info(f"✅ Retrieved {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting MT5 data: {e}")
            return pd.DataFrame()
        finally:
            self.shutdown_mt5()
    
    def analyze_smc_on_mt5_data(self, symbol: str, timeframe: str, profile: str = 'swing') -> Dict[str, Any]:
        """Run SMC analysis on MT5 data"""
        # Get data from MT5
        df = self.get_mt5_data(symbol, timeframe, 500)
        
        if df.empty:
            return {'error': f'No MT5 data available for {symbol}'}
        
        # Convert symbol for SMC analysis
        smc_symbol = _match_symbol_key(symbol)
        
        # Run SMC analysis
        try:
            context = get_market_context_smc(df, profile=profile, symbol=smc_symbol)
            context['data_source'] = 'MT5'
            context['mt5_symbol'] = symbol
            context['analysis_time'] = datetime.now()
            
            return context
            
        except Exception as e:
            self.logger.error(f"SMC analysis error for {symbol}: {e}")
            return {'error': f'SMC analysis failed: {str(e)}'}
    
    def generate_smc_signal(self, symbol: str, timeframe: str, profile: str = 'swing') -> Dict[str, Any]:
        """Generate trading signal based on SMC analysis"""
        context = self.analyze_smc_on_mt5_data(symbol, timeframe, profile)
        
        if 'error' in context:
            return context
        
        # Extract signal information
        entry_exit = context['entry_exit_analysis']
        smc_analysis = context['smc_analysis']
        current_price = context['current_price']
        atr = context['atr']
        
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'profile': profile,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'market_bias': context['market_bias'],
            'bias_strength': context['bias_strength'],
            'direction': entry_exit['direction'],
            'risk_level': context['risk_assessment']['overall_risk'],
            'smc_elements': {
                'structure_type': smc_analysis['structure']['structure_type'],
                'current_zone': smc_analysis['premium_discount_zones']['current_zone'],
                'active_order_blocks': len(smc_analysis['order_blocks']),
                'active_fvgs': len(smc_analysis['fair_value_gaps']),
                'liquidity_zones': len(smc_analysis['liquidity_zones'])
            }
        }
        
        # Generate entry details if signal is valid
        if entry_exit['direction'] != 'WAIT' and entry_exit['entry_zones']:
            best_zone = entry_exit['entry_zones'][0]
            
            signal.update({
                'entry_price': best_zone['entry_price'],
                'entry_zone_type': best_zone['type'],
                'confidence': best_zone['confidence'],
                'stop_loss': best_zone.get('stop_loss'),
                'take_profit': best_zone.get('take_profit')
            })
            
            # Calculate position size based on risk
            if signal['stop_loss']:
                risk_amount = self.calculate_position_size(
                    symbol, signal['entry_price'], signal['stop_loss']
                )
                signal['position_size'] = risk_amount
        
        return signal
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        if not self.initialize_mt5():
            return 0.01  # Default minimum lot
        
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return 0.01
            
            balance = account_info.balance
            risk_amount = balance * self.risk_settings['max_risk_per_trade']
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Calculate pip value
            if 'JPY' in symbol:
                pip_size = 0.01
            else:
                pip_size = 0.0001
            
            # Risk per pip
            risk_pips = abs(entry_price - stop_loss) / pip_size
            
            if risk_pips > 0:
                pip_value = symbol_info.trade_tick_value
                position_size = risk_amount / (risk_pips * pip_value)
                
                # Round to valid lot size
                min_lot = symbol_info.volume_min
                lot_step = symbol_info.volume_step
                
                position_size = max(min_lot, round(position_size / lot_step) * lot_step)
                return min(position_size, symbol_info.volume_max)
            
            return 0.01
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01
        finally:
            self.shutdown_mt5()
    
    def place_mt5_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Place order in MT5 based on SMC signal"""
        if not self.initialize_mt5():
            return {'success': False, 'error': 'MT5 initialization failed'}
        
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal.get('entry_price', signal['current_price'])
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            volume = signal.get('position_size', 0.01)
            
            # Get current price for order type determination
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {'success': False, 'error': f'Cannot get tick data for {symbol}'}
            
            current_price = tick.bid if direction == 'SELL' else tick.ask
            
            # Determine order type
            if direction == 'BUY':
                if entry_price <= current_price + 0.0001:  # Market or very close
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                elif entry_price < current_price:
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT
                    price = entry_price
                else:
                    order_type = mt5.ORDER_TYPE_BUY_STOP
                    price = entry_price
            else:  # SELL
                if entry_price >= current_price - 0.0001:  # Market or very close
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                elif entry_price > current_price:
                    order_type = mt5.ORDER_TYPE_SELL_LIMIT
                    price = entry_price
                else:
                    order_type = mt5.ORDER_TYPE_SELL_STOP
                    price = entry_price
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_PENDING if order_type in [
                    mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT,
                    mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP
                ] else mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 10,
                "magic": 100234,  # SMC Magic Number
                "comment": f"SMC_{signal['profile']}_{signal['entry_zone_type']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if available
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Order failed: {result.retcode}',
                    'result': result._asdict()
                }
            
            self.logger.info(f"✅ SMC order placed: {symbol} {direction} @ {price}")
            
            return {
                'success': True,
                'ticket': result.order,
                'price': result.price,
                'volume': result.volume,
                'result': result._asdict()
            }
            
        except Exception as e:
            self.logger.error(f"Error placing MT5 order: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self.shutdown_mt5()
    
    def broadcast_smc_signal(self, signal: Dict[str, Any], server_config: Dict[str, str]) -> bool:
        """Broadcast SMC signal to app_rev4.py server"""
        try:
            # Format signal for broadcasting
            broadcast_data = {
                'api_key': server_config.get('api_key'),
                'secret_key': server_config.get('secret_key'),
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'entry_price': signal.get('entry_price', signal['current_price']),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence', 0.0),
                'analysis_type': 'SMC',
                'profile': signal['profile'],
                'bias': signal['market_bias'],
                'risk_level': signal['risk_level'],
                'smc_context': {
                    'structure_type': signal['smc_elements']['structure_type'],
                    'current_zone': signal['smc_elements']['current_zone'],
                    'entry_zone_type': signal.get('entry_zone_type', ''),
                    'bias_strength': signal['bias_strength']
                },
                'timestamp': signal['timestamp'].isoformat()
            }
            
            # Convert to signal format expected by app_rev4.py
            signal_json = self._format_for_broadcast(broadcast_data)
            
            # Send to server
            response = requests.post(
                server_config['server_url'],
                json={
                    'api_key': server_config['api_key'],
                    'signal_json': signal_json,
                    'secret_key': server_config['secret_key']
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"✅ SMC signal broadcast successfully")
                return True
            else:
                self.logger.error(f"Signal broadcast failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error broadcasting signal: {e}")
            return False
    
    def _format_for_broadcast(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Format signal data for app_rev4.py broadcast system"""
        signal_json = {
            "Symbol": data['symbol'],
            "BuyEntry": "", "BuySL": "", "BuyTP": "",
            "SellEntry": "", "SellSL": "", "SellTP": "",
            "BuyLimit": "", "BuyLimitSL": "", "BuyLimitTP": "",
            "SellLimit": "", "SellLimitSL": "", "SellLimitTP": "",
            "BuyStop": "", "BuyStopSL": "", "BuyStopTP": "",
            "SellStop": "", "SellStopSL": "", "SellStopTP": "",
            "DeleteLimitStop": "",
            # SMC specific fields
            "SMC_Bias": data['bias'],
            "SMC_Profile": data['profile'],
            "SMC_Zone": data['smc_context']['current_zone'],
            "SMC_Structure": data['smc_context']['structure_type'],
            "SMC_Confidence": str(data['confidence'])
        }
        
        direction = data['direction']
        entry = str(data['entry_price'])
        sl = str(data['stop_loss']) if data['stop_loss'] else ""
        tp = str(data['take_profit']) if data['take_profit'] else ""
        
        if direction == 'BUY':
            signal_json.update({
                "BuyEntry": entry,
                "BuySL": sl,
                "BuyTP": tp
            })
        elif direction == 'SELL':
            signal_json.update({
                "SellEntry": entry,
                "SellSL": sl,
                "SellTP": tp
            })
        
        return signal_json
    
    def run_smc_scanner(self, symbols: List[str], timeframes: List[str], 
                       profiles: List[str], server_config: Dict[str, str],
                       scan_interval: int = 300) -> None:
        """Run continuous SMC scanning and signal generation"""
        self.logger.info(f"🔍 Starting SMC Scanner for {len(symbols)} symbols")
        
        while True:
            try:
                for symbol in symbols:
                    for timeframe in timeframes:
                        for profile in profiles:
                            self.logger.info(f"📊 Scanning {symbol} {timeframe} {profile}")
                            
                            # Generate SMC signal
                            signal = self.generate_smc_signal(symbol, timeframe, profile)
                            
                            if 'error' not in signal and signal['direction'] != 'WAIT':
                                self.logger.info(f"🎯 SMC Signal: {symbol} {signal['direction']} - {signal['market_bias']}")
                                
                                # Check if this is a new signal (avoid duplicates)
                                signal_key = f"{symbol}_{timeframe}_{profile}_{signal['direction']}"
                                
                                if self._is_new_signal(signal_key, signal):
                                    # Place order in MT5
                                    if server_config.get('auto_trade', False):
                                        order_result = self.place_mt5_order(signal)
                                        signal['mt5_order'] = order_result
                                    
                                    # Broadcast signal
                                    if server_config.get('broadcast', True):
                                        self.broadcast_smc_signal(signal, server_config)
                                    
                                    # Save signal
                                    self.last_signals[signal_key] = signal
                                    
                            elif 'error' in signal:
                                self.logger.error(f"❌ {symbol} {timeframe}: {signal['error']}")
                            
                            # Small delay between analyses
                            time.sleep(1)
                
                # Wait for next scan cycle
                self.logger.info(f"⏳ Waiting {scan_interval} seconds for next scan...")
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 SMC Scanner stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in SMC scanner: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _is_new_signal(self, signal_key: str, new_signal: Dict[str, Any]) -> bool:
        """Check if this is a new signal to avoid duplicates"""
        if signal_key not in self.last_signals:
            return True
        
        last_signal = self.last_signals[signal_key]
        time_diff = new_signal['timestamp'] - last_signal['timestamp']
        
        # Consider new if more than 30 minutes old or different entry price
        if time_diff > timedelta(minutes=30):
            return True
        
        if abs(new_signal.get('entry_price', 0) - last_signal.get('entry_price', 0)) > 0.001:
            return True
        
        return False

# ===============================
# Configuration and Usage Example
# ===============================

def create_smc_config() -> Dict[str, Any]:
    """Create SMC scanner configuration"""
    return {
        'symbols': ['XAUUSD', 'EURUSD', 'GBPUSD', 'BTCUSD'],
        'timeframes': ['15m', '1h', '4h'],
        'profiles': ['scalper', 'intraday', 'swing'],
        'scan_interval': 300,  # 5 minutes
        'server_config': {
            'server_url': 'http://localhost:5000/submit_signal',
            'api_key': 'your_api_key_here',
            'secret_key': 'your_secret_key_here',
            'auto_trade': False,  # Set to True for auto-trading
            'broadcast': True
        },
        'mt5_path': 'C:/Program Files/MetaTrader 5/terminal64.exe'
    }

if __name__ == "__main__":
    # Example usage
    config = create_smc_config()
    
    smc_mt5 = SMC_MT5_Integration(config['mt5_path'])
    
    print("🎯 SMC-MT5 Integration System")
    print("=" * 50)
    
    # Test single analysis
    print("\n📊 Testing single SMC analysis...")
    signal = smc_mt5.generate_smc_signal('XAUUSD', '1h', 'swing')
    
    if 'error' not in signal:
        print(f"✅ Signal generated:")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Direction: {signal['direction']}")
        print(f"   Bias: {signal['market_bias']}")
        print(f"   Risk: {signal['risk_level']}")
        
        if signal['direction'] != 'WAIT':
            print(f"   Entry: {signal.get('entry_price', 'N/A')}")
            print(f"   SL: {signal.get('stop_loss', 'N/A')}")
            print(f"   TP: {signal.get('take_profit', 'N/A')}")
    else:
        print(f"❌ Error: {signal['error']}")
    
    print("\n🔍 To start continuous scanning, run:")
    print("smc_mt5.run_smc_scanner(**config)")
