#!/usr/bin/env python3
# main_smc.py
# Pure Smart Money Concepts (SMC) Trading System - Main Application
# Author: AI Assistant
# Version: 1.0

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smc_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import our SMC system
try:
    from analyze_market_smc import get_market_context_smc, fetch_data, backtest_smc_strategy
    from technical_indicators_smc import analyze_smc_full
    print("✅ SMC Trading System modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing SMC modules: {e}")
    sys.exit(1)

class SMCTradingApp:
    """Main SMC Trading Application"""
    
    def __init__(self):
        self.version = "1.0"
        self.app_name = "Pure SMC Trading System"
        self.supported_symbols = {
            'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
            'metals': ['GC=F', 'SI=F'],
            'indices': ['^GSPC', '^DJI', '^IXIC']
        }
        self.profiles = ['scalper', 'intraday', 'swing', 'position']
        
    def display_banner(self):
        """Display application banner"""
        print("=" * 70)
        print(f"🎯 {self.app_name} v{self.version}")
        print("Pure Smart Money Concepts Analysis - No Traditional Indicators")
        print("=" * 70)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def display_menu(self):
        """Display main menu options"""
        print("📋 Available Options:")
        print("1. 🔍 Analyze Single Symbol")
        print("2. 📊 Multi-Symbol Dashboard")
        print("3. 📈 Run Backtesting")
        print("4. 🎯 Demo Analysis")
        print("5. 🧪 Test System Components")
        print("6. 📚 Show Documentation")
        print("7. 🚪 Exit")
        print()

    def display_supported_symbols(self):
        """Display supported symbols by category"""
        print("💎 Supported Symbols by Category:")
        for category, symbols in self.supported_symbols.items():
            print(f"  📂 {category.title()}: {', '.join(symbols)}")
        print()

    def analyze_single_symbol(self):
        """Analyze a single symbol with SMC"""
        print("\n🔍 Single Symbol SMC Analysis")
        print("-" * 50)
        
        # Get user input
        symbol = input("Enter symbol (e.g., BTC-USD, GC=F, EURUSD=X): ").strip().upper()
        
        print("👤 Available Profiles:")
        for i, profile in enumerate(self.profiles, 1):
            print(f"  {i}. {profile.title()}")
        
        try:
            profile_choice = int(input("Select profile (1-4): ")) - 1
            if profile_choice < 0 or profile_choice >= len(self.profiles):
                raise ValueError("Invalid profile selection")
            profile = self.profiles[profile_choice]
        except (ValueError, IndexError):
            print("❌ Invalid selection, using 'swing' profile")
            profile = 'swing'

        period = input("Data period (default: 1mo): ").strip() or '1mo'
        interval = input("Interval (default: 4h): ").strip() or '4h'
        
        print(f"\n🔄 Fetching data for {symbol}...")
        
        # Fetch data
        df = fetch_data(symbol, period=period, interval=interval)
        
        if df.empty:
            print(f"❌ No data available for {symbol}")
            return
        
        print(f"✅ Loaded {len(df)} candles")
        print(f"📅 Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Run SMC analysis
        print(f"\n🎯 Running SMC Analysis ({profile.title()} Profile)...")
        
        # Convert symbol for internal use
        internal_symbol = self.convert_symbol_for_analysis(symbol)
        context = get_market_context_smc(df, profile=profile, symbol=internal_symbol)
        
        if 'error' in context:
            print(f"❌ Analysis Error: {context['error']}")
            return
        
        # Display results
        self.display_analysis_results(context, symbol, profile)
        
        # Ask for backtesting
        if input("\n📈 Run backtesting? (y/n): ").lower().startswith('y'):
            self.run_backtest_for_symbol(df, context, internal_symbol)

    def convert_symbol_for_analysis(self, symbol):
        """Convert yahoo finance symbol to internal symbol format"""
        conversions = {
            'BTC-USD': 'BTCUSD',
            'ETH-USD': 'ETHUSD', 
            'GC=F': 'XAUUSD',
            'SI=F': 'XAGUSD',
            'EURUSD=X': 'EURUSD',
            'GBPUSD=X': 'GBPUSD',
            'USDJPY=X': 'USDJPY'
        }
        return conversions.get(symbol, 'DEFAULT')

    def display_analysis_results(self, context, symbol, profile):
        """Display comprehensive analysis results"""
        print(f"\n🎯 SMC Analysis Results for {symbol}")
        print("=" * 60)
        
        # Basic info
        print(f"Current Price: ${context['current_price']:.4f}")
        print(f"Market Bias: {context['market_bias']} (Strength: {context['bias_strength']})")
        print(f"ATR: ${context['atr']:.4f}")
        print(f"Overall Risk: {context['risk_assessment']['overall_risk']}")
        
        # SMC Elements
        smc = context['smc_analysis']
        print(f"\n📋 SMC Elements:")
        print(f"  Structure Type: {smc['structure']['structure_type']}")
        print(f"  Trend Direction: {smc['structure']['trend_direction']}")
        print(f"  Current Zone: {smc['premium_discount_zones']['current_zone']}")
        print(f"  Active Order Blocks: {len(smc['order_blocks'])}")
        print(f"  Active FVGs: {len(smc['fair_value_gaps'])}")
        print(f"  Liquidity Zones: {len(smc['liquidity_zones'])}")
        
        # Structure signals
        if smc['structure']['bos_detected']:
            print(f"  🚨 BOS Detected: {smc['structure']['trend_direction']}")
        
        if smc['change_of_character']:
            choch = smc['change_of_character']
            print(f"  🔄 CHoCH: {choch['previous_trend']} → {choch['new_trend']}")
        
        # Liquidity sweeps
        if smc['liquidity_sweeps']:
            print(f"  💧 Liquidity Sweeps: {len(smc['liquidity_sweeps'])}")
            for sweep in smc['liquidity_sweeps'][:2]:
                print(f"     - {sweep['direction']} sweep at ${sweep['swept_price']:.4f}")
        
        # Entry recommendation
        entry_exit = context['entry_exit_analysis']
        print(f"\n💡 Trading Recommendation: {entry_exit['direction']}")
        
        if entry_exit['direction'] != 'WAIT' and entry_exit['entry_zones']:
            best_zone = entry_exit['entry_zones'][0]
            print(f"  Best Entry Zone: {best_zone['type']}")
            print(f"  Entry Price: ${best_zone['entry_price']:.4f}")
            if 'stop_loss' in best_zone:
                print(f"  Stop Loss: ${best_zone['stop_loss']:.4f}")
            if 'take_profit' in best_zone:
                print(f"  Take Profit: ${best_zone['take_profit']:.4f}")
            print(f"  Confidence: {best_zone['confidence']:.2f}")
        
        # Profile-specific analysis
        if context.get('profile_analysis'):
            profile_data = context['profile_analysis']
            print(f"\n👤 {profile.title()} Profile Analysis:")
            print(f"  Market Phase: {profile_data['market_phase']}")
            print(f"  Confidence: {profile_data['confidence']:.1%}")
            print(f"  Entry Signals: {len(profile_data['entry_signals'])}")
            
            if profile_data.get('warnings'):
                print(f"  ⚠️ Warnings: {', '.join(profile_data['warnings'])}")
            
            # Show entry signals
            for i, signal in enumerate(profile_data['entry_signals'][:3], 1):
                print(f"  Signal {i}: {signal['type']} ({signal.get('timeframe', 'N/A')})")

    def multi_symbol_dashboard(self):
        """Display multi-symbol dashboard"""
        print("\n📊 Multi-Symbol SMC Dashboard")
        print("-" * 50)
        
        # Select symbols to analyze
        default_symbols = ['BTC-USD', 'GC=F', 'EURUSD=X']
        symbols_input = input(f"Enter symbols separated by comma (default: {','.join(default_symbols)}): ").strip()
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = default_symbols
        
        profile = input("Profile (scalper/intraday/swing/position, default: swing): ").strip() or 'swing'
        
        print(f"\n🔄 Analyzing {len(symbols)} symbols with {profile} profile...")
        
        results = []
        
        for symbol in symbols:
            print(f"\n📈 Analyzing {symbol}...")
            
            # Fetch data
            df = fetch_data(symbol, period='1mo', interval='4h')
            
            if df.empty:
                print(f"❌ No data for {symbol}")
                continue
            
            # Run analysis
            internal_symbol = self.convert_symbol_for_analysis(symbol)
            context = get_market_context_smc(df, profile=profile, symbol=internal_symbol)
            
            if 'error' in context:
                print(f"❌ Error analyzing {symbol}: {context['error']}")
                continue
            
            # Store results
            result = {
                'symbol': symbol,
                'price': context['current_price'],
                'bias': context['market_bias'],
                'strength': context['bias_strength'],
                'zone': context['smc_analysis']['premium_discount_zones']['current_zone'],
                'risk': context['risk_assessment']['overall_risk'],
                'direction': context['entry_exit_analysis']['direction'],
                'entry_zones': len(context['entry_exit_analysis']['entry_zones'])
            }
            results.append(result)
        
        # Display dashboard
        print(f"\n📊 SMC Dashboard ({profile.title()} Profile)")
        print("=" * 100)
        print(f"{'Symbol':<12} {'Price':<12} {'Bias':<16} {'Zone':<12} {'Risk':<8} {'Direction':<8} {'Zones':<6}")
        print("-" * 100)
        
        for result in results:
            print(f"{result['symbol']:<12} "
                  f"${result['price']:<11.4f} "
                  f"{result['bias']:<16} "
                  f"{result['zone']:<12} "
                  f"{result['risk']:<8} "
                  f"{result['direction']:<8} "
                  f"{result['entry_zones']:<6}")
        
        # Summary
        total_symbols = len(results)
        bullish_count = len([r for r in results if 'BULLISH' in r['bias']])
        bearish_count = len([r for r in results if 'BEARISH' in r['bias']])
        
        print(f"\n📋 Dashboard Summary:")
        print(f"  Total Symbols: {total_symbols}")
        
        if total_symbols > 0:
            print(f"  Bullish Bias: {bullish_count} ({bullish_count/total_symbols*100:.1f}%)")
            print(f"  Bearish Bias: {bearish_count} ({bearish_count/total_symbols*100:.1f}%)")
            print(f"  Entry Opportunities: {len([r for r in results if r['direction'] != 'WAIT'])}")
        else:
            print("  No symbols successfully analyzed. Check symbol formats and network connection.")

    def run_backtest_for_symbol(self, df, context, symbol):
        """Run backtesting for a specific symbol"""
        print(f"\n📈 Running SMC Backtesting for {symbol}...")
        
        # Get backtesting parameters
        try:
            capital = float(input("Initial Capital (default: 10000): ") or 10000)
            risk = float(input("Risk per trade % (default: 2): ") or 2) / 100
        except ValueError:
            capital = 10000
            risk = 0.02
        
        print(f"🔄 Backtesting with ${capital:,.2f} capital, {risk:.1%} risk per trade...")
        
        # Run backtest
        results = backtest_smc_strategy(df, context, 
                                       initial_capital=capital,
                                       risk_per_trade=risk,
                                       symbol=symbol)
        
        if 'error' in results:
            print(f"❌ Backtest Error: {results['error']}")
            return
        
        # Display results
        print(f"\n📊 Backtesting Results:")
        print(f"  Total PnL: ${results['total_pnl']:,.2f}")
        print(f"  Final Equity: ${results['final_equity']:,.2f}")
        print(f"  Return: {((results['final_equity']/capital)-1)*100:+.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Max Drawdown: ${results['max_drawdown']:,.2f}")
        
        # SMC Strategy breakdown
        if 'smc_strategy_breakdown' in results:
            breakdown = results['smc_strategy_breakdown']
            print(f"\n🎯 SMC Strategy Breakdown:")
            print(f"  Order Block Trades: {breakdown['order_block_trades']}")
            print(f"  FVG Trades: {breakdown['fvg_trades']}")
            print(f"  Zone Trades: {breakdown['zone_trades']}")

    def demo_analysis(self):
        """Run demo analysis with pre-selected data"""
        print("\n🎯 SMC Analysis Demo")
        print("-" * 50)
        
        # Demo with Bitcoin
        print("📊 Demo: Bitcoin (BTC-USD) Analysis")
        
        try:
            df = fetch_data('BTC-USD', period='1mo', interval='4h')
            
            if df.empty:
                print("❌ Demo failed: No data available")
                return
            
            context = get_market_context_smc(df, profile='swing', symbol='BTCUSD')
            
            if 'error' in context:
                print(f"❌ Demo failed: {context['error']}")
                return
            
            self.display_analysis_results(context, 'BTC-USD', 'swing')
            
            # Quick backtest
            print(f"\n📈 Demo Backtesting...")
            results = backtest_smc_strategy(df, context, symbol='BTCUSD')
            
            if 'error' not in results:
                print(f"  Demo PnL: ${results['total_pnl']:,.2f}")
                print(f"  Win Rate: {results['win_rate']:.1%}")
                print(f"  Profit Factor: {results['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"❌ Demo error: {str(e)}")

    def test_system_components(self):
        """Test all system components"""
        print("\n🧪 Testing SMC System Components")
        print("-" * 50)
        
        from simple_test_smc import test_smc_components, create_dummy_data
        
        # Create test data
        print("🔄 Creating test data...")
        df = create_dummy_data(200)
        print(f"✅ Generated {len(df)} test candles")
        
        # Test components
        test_smc_components()
        
        # Test full analysis
        print("\n🔄 Testing full SMC analysis...")
        try:
            context = get_market_context_smc(df, profile='swing', symbol='BTCUSD')
            if 'error' not in context:
                print(f"✅ Full analysis test passed")
                print(f"  Market Bias: {context['market_bias']}")
                print(f"  Entry Direction: {context['entry_exit_analysis']['direction']}")
            else:
                print(f"❌ Full analysis test failed: {context['error']}")
        except Exception as e:
            print(f"❌ Full analysis test error: {str(e)}")

    def show_documentation(self):
        """Display system documentation"""
        print("\n📚 SMC Trading System Documentation")
        print("-" * 50)
        
        doc_text = """
🎯 Pure Smart Money Concepts (SMC) Trading System

📋 Core Components:
  • Market Structure Analysis (BOS/CHoCH)
  • Liquidity Zone Detection (BSL/SSL)
  • Order Block Identification
  • Fair Value Gap Analysis
  • Premium/Discount Zones

👤 Trading Profiles:
  • Scalper: M1-M5, quick setups
  • Intraday: M15-H1, session awareness
  • Swing: H1-H4, structure focus
  • Position: H4-D1, long-term view

🎯 Key SMC Signals:
  • BOS: Trend continuation
  • CHoCH: Trend reversal
  • Liquidity Sweep: Smart money action
  • Order Block Test: Reaction zones
  • FVG Fill: Magnetic zones

📈 Usage:
  1. Select symbol and profile
  2. System analyzes market structure
  3. Provides entry/exit recommendations
  4. Includes risk assessment
  5. Optional backtesting validation

⚠️ Risk Management:
  • Always use stop losses
  • Follow risk assessment warnings
  • Position size based on risk %
  • Validate spread conditions
        """
        print(doc_text)

    def run(self):
        """Main application loop"""
        self.display_banner()
        
        while True:
            try:
                self.display_menu()
                choice = input("Select option (1-7): ").strip()
                
                if choice == '1':
                    self.analyze_single_symbol()
                elif choice == '2':
                    self.multi_symbol_dashboard()
                elif choice == '3':
                    symbol = input("Symbol for backtesting (e.g., BTC-USD): ").strip().upper()
                    if symbol:
                        df = fetch_data(symbol, period='2mo', interval='4h')
                        if not df.empty:
                            internal_symbol = self.convert_symbol_for_analysis(symbol)
                            context = get_market_context_smc(df, profile='swing', symbol=internal_symbol)
                            if 'error' not in context:
                                self.run_backtest_for_symbol(df, context, internal_symbol)
                elif choice == '4':
                    self.demo_analysis()
                elif choice == '5':
                    self.test_system_components()
                elif choice == '6':
                    self.show_documentation()
                elif choice == '7':
                    print("\n👋 Thank you for using SMC Trading System!")
                    print("🎯 Happy SMC Trading! 📈")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                
                input("\n⏸️ Press Enter to continue...")
                print("\n" + "="*70)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Application interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                logging.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Initialize and run the SMC Trading Application
    try:
        app = SMCTradingApp()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        logging.error(f"Application startup error: {str(e)}", exc_info=True)
        sys.exit(1)
