#!/usr/bin/env python3
# main_smc_mt5.py
# Main application untuk SMC Trading System dengan integrasi MT5 dan broadcasting
# Menghubungkan sistem SMC dengan MetaTrader 5 dan signal broadcasting ke app_rev4.py

import sys
import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any

# Import SMC-MT5 integration
from smc_mt5_integration import SMC_MT5_Integration, create_smc_config

class SMCTradingBot:
    """Main SMC Trading Bot with MT5 integration and signal broadcasting"""
    
    def __init__(self, config_file: str = "smc_config.json"):
        self.config_file = config_file
        
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('smc_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SMCTradingBot')
        
        # Then load config and initialize other components
        self.config = self.load_config()
        self.smc_mt5 = SMC_MT5_Integration(self.config.get('mt5_path', ''))
        self.running = False
        self.scanner_thread = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"✅ Configuration loaded from {self.config_file}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        
        # Create default config
        config = create_smc_config()
        self.save_config(config)
        self.logger.info(f"📄 Default configuration created: {self.config_file}")
        return config
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4, default=str)
            self.logger.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def display_banner(self):
        """Display application banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                    🎯 SMC Trading Bot v2.0                       ║
║               Smart Money Concepts + MT5 Integration              ║
║                   With Signal Broadcasting                        ║
╚═══════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def display_menu(self):
        """Display main menu"""
        print("📋 Available Options:")
        print("1. 🔍 Test SMC Analysis (Single Symbol)")
        print("2. 📊 Run Market Scanner")
        print("3. 🤖 Start Auto Trading Bot")
        print("4. ⚙️  Configure Settings")
        print("5. 📈 View Trading Statistics")
        print("6. 🧪 Test MT5 Connection")
        print("7. 📡 Test Signal Broadcasting")
        print("8. 🚪 Exit")
        print()
    
    def test_single_analysis(self):
        """Test SMC analysis on single symbol"""
        print("\n🔍 SMC Analysis Test")
        print("-" * 50)
        
        # Get user input
        symbol = input("Enter symbol (default: XAUUSD): ").strip() or 'XAUUSD'
        timeframe = input("Enter timeframe (default: 1h): ").strip() or '1h'
        profile = input("Enter profile (scalper/intraday/swing/position, default: swing): ").strip() or 'swing'
        
        print(f"\n🔄 Analyzing {symbol} {timeframe} with {profile} profile...")
        
        # Run analysis
        signal = self.smc_mt5.generate_smc_signal(symbol, timeframe, profile)
        
        if 'error' in signal:
            print(f"❌ Analysis Error: {signal['error']}")
            return
        
        # Display results
        print(f"\n📊 SMC Analysis Results:")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Current Price: ${signal['current_price']:.5f}")
        print(f"   Market Bias: {signal['market_bias']} (Strength: {signal['bias_strength']})")
        print(f"   Direction: {signal['direction']}")
        print(f"   Risk Level: {signal['risk_level']}")
        
        # SMC Elements
        smc_elements = signal['smc_elements']
        print(f"\n🧩 SMC Elements:")
        print(f"   Structure Type: {smc_elements['structure_type']}")
        print(f"   Current Zone: {smc_elements['current_zone']}")
        print(f"   Active Order Blocks: {smc_elements['active_order_blocks']}")
        print(f"   Active FVGs: {smc_elements['active_fvgs']}")
        print(f"   Liquidity Zones: {smc_elements['liquidity_zones']}")
        
        # Entry details if available
        if signal['direction'] != 'WAIT':
            print(f"\n💡 Trading Signal:")
            print(f"   Entry Price: ${signal.get('entry_price', 'N/A')}")
            print(f"   Entry Zone: {signal.get('entry_zone_type', 'N/A')}")
            print(f"   Stop Loss: ${signal.get('stop_loss', 'N/A')}")
            print(f"   Take Profit: ${signal.get('take_profit', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 0):.2f}")
            print(f"   Position Size: {signal.get('position_size', 'N/A')} lots")
            
            # Ask if user wants to broadcast or place order
            choice = input("\n📡 Broadcast this signal? (y/n): ").lower().strip()
            if choice.startswith('y'):
                success = self.smc_mt5.broadcast_smc_signal(signal, self.config['server_config'])
                if success:
                    print("✅ Signal broadcast successfully!")
                else:
                    print("❌ Failed to broadcast signal")
            
            choice = input("🤖 Place order in MT5? (y/n): ").lower().strip()
            if choice.startswith('y'):
                result = self.smc_mt5.place_mt5_order(signal)
                if result['success']:
                    print(f"✅ Order placed! Ticket: {result['ticket']}")
                else:
                    print(f"❌ Order failed: {result['error']}")
    
    def run_scanner(self):
        """Run market scanner"""
        print("\n📊 Market Scanner")
        print("-" * 50)
        
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']
        profiles = self.config['profiles']
        
        print(f"🔍 Scanning Configuration:")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Timeframes: {', '.join(timeframes)}")
        print(f"   Profiles: {', '.join(profiles)}")
        print(f"   Scan Interval: {self.config['scan_interval']} seconds")
        
        choice = input("\n🚀 Start scanner? (y/n): ").lower().strip()
        if not choice.startswith('y'):
            return
        
        print("\n📡 Starting SMC Scanner...")
        print("Press Ctrl+C to stop")
        
        try:
            self.smc_mt5.run_smc_scanner(
                symbols=symbols,
                timeframes=timeframes,
                profiles=profiles,
                server_config=self.config['server_config'],
                scan_interval=self.config['scan_interval']
            )
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
    
    def start_auto_trading(self):
        """Start automated trading bot"""
        print("\n🤖 Auto Trading Bot")
        print("-" * 50)
        
        if not self.config['server_config'].get('auto_trade', False):
            print("⚠️  Auto trading is disabled in configuration")
            choice = input("Enable auto trading? (y/n): ").lower().strip()
            if choice.startswith('y'):
                self.config['server_config']['auto_trade'] = True
                self.save_config(self.config)
                print("✅ Auto trading enabled")
            else:
                return
        
        print("🚨 WARNING: Auto trading will place real orders in MT5!")
        print("Make sure you understand the risks involved.")
        confirmation = input("Type 'CONFIRM' to start auto trading: ").strip()
        
        if confirmation != 'CONFIRM':
            print("❌ Auto trading cancelled")
            return
        
        print("\n🤖 Starting Auto Trading Bot...")
        print("Press Ctrl+C to stop")
        
        # Start scanner in auto-trade mode
        self.running = True
        self.scanner_thread = threading.Thread(target=self._auto_trading_loop)
        self.scanner_thread.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping auto trading...")
            self.running = False
            if self.scanner_thread:
                self.scanner_thread.join()
            print("✅ Auto trading stopped")
    
    def _auto_trading_loop(self):
        """Auto trading loop running in separate thread"""
        self.smc_mt5.run_smc_scanner(
            symbols=self.config['symbols'],
            timeframes=self.config['timeframes'],
            profiles=self.config['profiles'],
            server_config=self.config['server_config'],
            scan_interval=self.config['scan_interval']
        )
    
    def configure_settings(self):
        """Configure bot settings"""
        print("\n⚙️ Configuration Settings")
        print("-" * 50)
        
        print("Current Configuration:")
        print(f"   MT5 Path: {self.config.get('mt5_path', 'Not set')}")
        print(f"   Symbols: {', '.join(self.config['symbols'])}")
        print(f"   Timeframes: {', '.join(self.config['timeframes'])}")
        print(f"   Profiles: {', '.join(self.config['profiles'])}")
        print(f"   Scan Interval: {self.config['scan_interval']} seconds")
        print(f"   Server URL: {self.config['server_config'].get('server_url', 'Not set')}")
        print(f"   Auto Trade: {self.config['server_config'].get('auto_trade', False)}")
        print(f"   Broadcast: {self.config['server_config'].get('broadcast', True)}")
        
        print("\nConfiguration Options:")
        print("1. Edit Symbols")
        print("2. Edit Timeframes")
        print("3. Edit Profiles")
        print("4. Edit Scan Interval")
        print("5. Edit Server Configuration")
        print("6. Edit MT5 Path")
        print("7. Back to Main Menu")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            symbols = input(f"Enter symbols (comma separated, current: {','.join(self.config['symbols'])}): ").strip()
            if symbols:
                self.config['symbols'] = [s.strip().upper() for s in symbols.split(',')]
                
        elif choice == '2':
            timeframes = input(f"Enter timeframes (comma separated, current: {','.join(self.config['timeframes'])}): ").strip()
            if timeframes:
                self.config['timeframes'] = [s.strip() for s in timeframes.split(',')]
                
        elif choice == '3':
            profiles = input(f"Enter profiles (comma separated, current: {','.join(self.config['profiles'])}): ").strip()
            if profiles:
                self.config['profiles'] = [s.strip() for s in profiles.split(',')]
                
        elif choice == '4':
            interval = input(f"Enter scan interval in seconds (current: {self.config['scan_interval']}): ").strip()
            if interval.isdigit():
                self.config['scan_interval'] = int(interval)
                
        elif choice == '5':
            print("\nServer Configuration:")
            url = input(f"Server URL (current: {self.config['server_config'].get('server_url', '')}): ").strip()
            if url:
                self.config['server_config']['server_url'] = url
                
            api_key = input(f"API Key (current: {self.config['server_config'].get('api_key', '')[:10]}...): ").strip()
            if api_key:
                self.config['server_config']['api_key'] = api_key
                
            secret_key = input(f"Secret Key (current: {self.config['server_config'].get('secret_key', '')[:10]}...): ").strip()
            if secret_key:
                self.config['server_config']['secret_key'] = secret_key
                
        elif choice == '6':
            mt5_path = input(f"MT5 Path (current: {self.config.get('mt5_path', '')}): ").strip()
            if mt5_path:
                self.config['mt5_path'] = mt5_path
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            self.save_config(self.config)
            print("✅ Configuration updated!")
    
    def view_statistics(self):
        """View trading statistics"""
        print("\n📈 Trading Statistics")
        print("-" * 50)
        print("This feature will be implemented in future versions.")
        print("It will show:")
        print("   • Total signals generated")
        print("   • Success rate by profile")
        print("   • Performance by symbol")
        print("   • P&L summary")
    
    def test_mt5_connection(self):
        """Test MT5 connection"""
        print("\n🧪 Testing MT5 Connection")
        print("-" * 50)
        
        if self.smc_mt5.initialize_mt5():
            print("✅ MT5 connection successful!")
            
            # Test data retrieval
            test_symbol = 'XAUUSD'
            df = self.smc_mt5.get_mt5_data(test_symbol, '1h', 100)
            
            if not df.empty:
                print(f"✅ Data retrieval successful: {len(df)} candles")
                print(f"   Latest price: ${df['close'].iloc[-1]:.5f}")
                print(f"   Data range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            else:
                print(f"❌ Failed to retrieve data for {test_symbol}")
        else:
            print("❌ MT5 connection failed!")
            print("Check MT5 path in configuration")
    
    def test_signal_broadcasting(self):
        """Test signal broadcasting"""
        print("\n📡 Testing Signal Broadcasting")
        print("-" * 50)
        
        # Create dummy signal for testing
        test_signal = {
            'symbol': 'XAUUSD',
            'timeframe': '1h',
            'profile': 'swing',
            'timestamp': datetime.now(),
            'current_price': 2000.00,
            'market_bias': 'BULLISH',
            'bias_strength': 5,
            'direction': 'BUY',
            'risk_level': 'MEDIUM',
            'entry_price': 1995.00,
            'entry_zone_type': 'ORDER_BLOCK',
            'confidence': 0.85,
            'stop_loss': 1990.00,
            'take_profit': 2010.00,
            'smc_elements': {
                'structure_type': 'BOS_BULLISH',
                'current_zone': 'DISCOUNT',
                'active_order_blocks': 2,
                'active_fvgs': 1,
                'liquidity_zones': 3
            }
        }
        
        print("🔄 Broadcasting test signal...")
        success = self.smc_mt5.broadcast_smc_signal(test_signal, self.config['server_config'])
        
        if success:
            print("✅ Signal broadcast test successful!")
        else:
            print("❌ Signal broadcast test failed!")
            print("Check server configuration and connectivity")
    
    def run(self):
        """Main application loop"""
        self.display_banner()
        
        while True:
            try:
                self.display_menu()
                choice = input("Select option (1-8): ").strip()
                
                if choice == '1':
                    self.test_single_analysis()
                elif choice == '2':
                    self.run_scanner()
                elif choice == '3':
                    self.start_auto_trading()
                elif choice == '4':
                    self.configure_settings()
                elif choice == '5':
                    self.view_statistics()
                elif choice == '6':
                    self.test_mt5_connection()
                elif choice == '7':
                    self.test_signal_broadcasting()
                elif choice == '8':
                    print("\n👋 Thank you for using SMC Trading Bot!")
                    print("🎯 Happy SMC Trading! 📈")
                    break
                else:
                    print("❌ Invalid option. Please select 1-8.")
                
                if choice != '8':
                    input("\n⏸️ Press Enter to continue...")
                    print("\n" + "="*70)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Application interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Application error: {str(e)}")
                print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    try:
        # Initialize and run the SMC Trading Bot
        bot = SMCTradingBot()
        bot.run()
    except Exception as e:
        print(f"❌ Failed to start SMC Trading Bot: {str(e)}")
        logging.error(f"Application startup error: {str(e)}", exc_info=True)
        sys.exit(1)
