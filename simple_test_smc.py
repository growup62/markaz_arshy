# simple_test_smc.py
# Simple test untuk SMC analysis menggunakan data dummy

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analyze_market_smc import get_market_context_smc, backtest_smc_strategy

def create_dummy_data(length=1000):
    """Create dummy OHLC data for testing"""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=length, freq='1H')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 50000  # Starting around $50k for BTC
    
    # Generate price movements
    returns = np.random.normal(0, 0.02, length)  # 2% volatility
    prices = [base_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        volatility = np.random.uniform(0.005, 0.015)  # 0.5-1.5% intraday volatility
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = price * (1 + np.random.uniform(-0.005, 0.005))
        
        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def test_smc_with_dummy_data():
    """Test SMC analysis with dummy data"""
    print("🔬 Testing SMC Analysis with Generated Data")
    print("=" * 60)
    
    # Create test data
    df = create_dummy_data(500)
    print(f"✅ Generated {len(df)} candles of test data")
    
    # Test different profiles
    profiles = ['scalper', 'intraday', 'swing', 'position']
    symbols = ['BTCUSD', 'XAUUSD', 'EURUSD']
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 40)
        
        for profile in profiles:
            try:
                context = get_market_context_smc(df, profile=profile, symbol=symbol)
                
                if 'error' in context:
                    print(f"❌ {profile.title()}: {context['error']}")
                    continue
                
                print(f"✅ {profile.title()}: {context['market_bias']} "
                      f"(Strength: {context['bias_strength']}) "
                      f"Zone: {context['smc_analysis']['premium_discount_zones']['current_zone']}")
                
                # Show key SMC elements
                smc = context['smc_analysis']
                print(f"   • Order Blocks: {len(smc['order_blocks'])}")
                print(f"   • FVGs: {len(smc['fair_value_gaps'])}")
                print(f"   • Liquidity Zones: {len(smc['liquidity_zones'])}")
                
                # Test entry recommendation
                entry_exit = context['entry_exit_analysis']
                if entry_exit['direction'] != 'WAIT':
                    print(f"   • Entry: {entry_exit['direction']} ({len(entry_exit['entry_zones'])} zones)")
                
            except Exception as e:
                print(f"❌ {profile.title()}: Error - {str(e)}")

def test_smc_components():
    """Test individual SMC components"""
    print("\n🔍 Testing Individual SMC Components")
    print("=" * 60)
    
    # Import SMC functions directly
    from technical_indicators_smc import (
        analyze_market_structure,
        detect_liquidity_zones,
        detect_order_blocks,
        detect_fair_value_gaps,
        calculate_premium_discount_zones
    )
    
    # Create test data
    df = create_dummy_data(200)
    
    # Test each component
    try:
        print("✅ Testing Market Structure...")
        structure = analyze_market_structure(df, swing_length=5)
        print(f"   Structure: {structure['structure_type']}")
        print(f"   Trend: {structure['trend_direction']}")
        print(f"   BOS Detected: {structure['bos_detected']}")
        
        print("✅ Testing Liquidity Zones...")
        liquidity = detect_liquidity_zones(df, lookback=20)
        print(f"   Found {len(liquidity)} liquidity zones")
        
        print("✅ Testing Order Blocks...")
        order_blocks = detect_order_blocks(df, lookback=50)
        print(f"   Found {len(order_blocks)} order blocks")
        
        print("✅ Testing Fair Value Gaps...")
        fvgs = detect_fair_value_gaps(df, min_gap_atr=0.1)
        print(f"   Found {len(fvgs)} FVGs")
        
        print("✅ Testing Premium/Discount Zones...")
        zones = calculate_premium_discount_zones(df)
        print(f"   Current Zone: {zones['current_zone']}")
        print(f"   Range: {zones['range_low']:.2f} - {zones['range_high']:.2f}")
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")

def test_backtesting():
    """Test SMC backtesting"""
    print("\n📈 Testing SMC Backtesting")
    print("=" * 60)
    
    # Create larger dataset for backtesting
    df = create_dummy_data(300)
    
    try:
        # Get SMC context
        context = get_market_context_smc(df, profile='swing', symbol='BTCUSD')
        
        if 'error' not in context:
            print("✅ Running backtest...")
            results = backtest_smc_strategy(
                df, 
                context, 
                initial_capital=10000,
                risk_per_trade=0.02,
                symbol='BTCUSD'
            )
            
            if 'error' not in results:
                print(f"✅ Backtest completed:")
                print(f"   • Total PnL: ${results['total_pnl']:.2f}")
                print(f"   • Total Trades: {results['total_trades']}")
                print(f"   • Win Rate: {results['win_rate']:.1%}")
                print(f"   • Profit Factor: {results.get('profit_factor', 0):.2f}")
                
                breakdown = results.get('smc_strategy_breakdown', {})
                print(f"   • Order Block Trades: {breakdown.get('order_block_trades', 0)}")
                print(f"   • FVG Trades: {breakdown.get('fvg_trades', 0)}")
                print(f"   • Zone Trades: {breakdown.get('zone_trades', 0)}")
            else:
                print(f"❌ Backtest failed: {results['error']}")
        else:
            print(f"❌ Context failed: {context['error']}")
            
    except Exception as e:
        print(f"❌ Backtesting failed: {str(e)}")

def demo_smc_analysis():
    """Demonstrate complete SMC analysis"""
    print("\n🎯 SMC Analysis Demo")
    print("=" * 60)
    
    # Create trending data
    np.random.seed(123)
    dates = pd.date_range(start=datetime(2023, 6, 1), periods=200, freq='4H')
    
    # Create uptrend data
    base_price = 2000  # Gold price
    trend_factor = 1.001  # Slight uptrend
    
    prices = [base_price]
    for i in range(1, 200):
        # Add trend and random walk
        change = np.random.normal(0, 0.01) + (trend_factor - 1)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLC
    data = []
    for i, price in enumerate(prices):
        vol = np.random.uniform(0.002, 0.008)
        data.append({
            'time': dates[i],
            'open': price * (1 + np.random.uniform(-0.002, 0.002)),
            'high': price * (1 + vol),
            'low': price * (1 - vol),
            'close': price,
            'volume': np.random.randint(500, 5000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"📊 Analyzing {len(df)} candles of Gold (XAUUSD) data")
    print(f"Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    try:
        # Comprehensive SMC analysis
        context = get_market_context_smc(df, profile='swing', symbol='XAUUSD')
        
        if 'error' not in context:
            print(f"\n🎯 SMC Analysis Results:")
            print(f"Current Price: ${context['current_price']:.2f}")
            print(f"Market Bias: {context['market_bias']} (Strength: {context['bias_strength']})")
            print(f"ATR: ${context['atr']:.2f}")
            print(f"Overall Risk: {context['risk_assessment']['overall_risk']}")
            
            # Detailed SMC breakdown
            smc = context['smc_analysis']
            print(f"\n📋 SMC Elements:")
            print(f"Structure Type: {smc['structure']['structure_type']}")
            print(f"Trend Direction: {smc['structure']['trend_direction']}")
            print(f"Current Zone: {smc['premium_discount_zones']['current_zone']}")
            
            if smc['change_of_character']:
                choch = smc['change_of_character']
                print(f"CHoCH Detected: {choch['previous_trend']} → {choch['new_trend']}")
            
            # Entry recommendations
            entry_exit = context['entry_exit_analysis']
            print(f"\n💡 Trading Recommendation: {entry_exit['direction']}")
            
            if entry_exit['direction'] != 'WAIT' and entry_exit['entry_zones']:
                best_zone = entry_exit['entry_zones'][0]
                print(f"Best Entry Zone: {best_zone['type']}")
                print(f"Entry Price: ${best_zone['entry_price']:.2f}")
                if 'stop_loss' in best_zone:
                    print(f"Stop Loss: ${best_zone['stop_loss']:.2f}")
                if 'take_profit' in best_zone:
                    print(f"Take Profit: ${best_zone['take_profit']:.2f}")
            
            # Profile-specific analysis
            if context.get('profile_analysis'):
                profile = context['profile_analysis']
                print(f"\n👤 Swing Trader Profile:")
                print(f"Market Phase: {profile['market_phase']}")
                print(f"Confidence: {profile['confidence']:.1%}")
                print(f"Entry Signals: {len(profile['entry_signals'])}")
        else:
            print(f"❌ Analysis failed: {context['error']}")
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run all tests
    test_smc_components()
    test_smc_with_dummy_data()
    test_backtesting()
    demo_smc_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 SMC Testing Complete!")
    print("=" * 60)
