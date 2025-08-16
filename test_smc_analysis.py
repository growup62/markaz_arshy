# test_smc_analysis.py
# Example script to test the new Pure SMC Analysis

from analyze_market_smc import get_market_context_smc, fetch_data, backtest_smc_strategy
import pandas as pd

def test_smc_analysis():
    """Test SMC analysis with different symbols and profiles"""
    
    # Test symbols
    test_cases = [
        {'ticker': 'BTC-USD', 'symbol': 'BTCUSD', 'profile': 'swing'},
        {'ticker': 'GC=F', 'symbol': 'XAUUSD', 'profile': 'intraday'},
        {'ticker': 'EURUSD=X', 'symbol': 'EURUSD', 'profile': 'scalper'}
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {case['ticker']} with {case['profile']} profile")
        print(f"{'='*60}")
        
        # Fetch data
        df = fetch_data(case['ticker'], period='2mo', interval='1h')
        
        if df.empty:
            print(f"❌ No data available for {case['ticker']}")
            continue
        
        # Run SMC Analysis
        context = get_market_context_smc(df, profile=case['profile'], symbol=case['symbol'])
        
        if 'error' in context:
            print(f"❌ Error: {context['error']}")
            continue
        
        # Display core results
        print(f"📊 Current Price: ${context['current_price']:.4f}")
        print(f"📈 Market Bias: {context['market_bias']} (Strength: {context['bias_strength']})")
        print(f"🎯 Current Zone: {context['smc_analysis']['premium_discount_zones']['current_zone']}")
        print(f"⚠️  Overall Risk: {context['risk_assessment']['overall_risk']}")
        
        # SMC Elements
        smc = context['smc_analysis']
        print(f"\n🔍 SMC Elements:")
        print(f"   • Active Order Blocks: {context['key_levels']['active_order_blocks']}")
        print(f"   • Active FVGs: {context['key_levels']['active_fvgs']}")
        print(f"   • Liquidity Zones: {context['key_levels']['liquidity_zones']}")
        print(f"   • Structure Type: {smc['structure']['structure_type']}")
        
        if smc['structure']['bos_detected']:
            print(f"   • 🚨 BOS Detected: {smc['structure']['trend_direction']}")
        
        if smc['change_of_character']:
            choch = smc['change_of_character']
            print(f"   • 🔄 CHoCH: {choch['previous_trend']} → {choch['new_trend']}")
        
        # Entry Analysis
        entry_exit = context['entry_exit_analysis']
        if entry_exit['direction'] != 'WAIT':
            print(f"\n💡 Entry Recommendation: {entry_exit['direction']}")
            if entry_exit['entry_zones']:
                best_zone = entry_exit['entry_zones'][0]
                print(f"   • Best Zone: {best_zone['type']}")
                print(f"   • Entry Price: ${best_zone['entry_price']:.4f}")
                print(f"   • Stop Loss: ${best_zone.get('stop_loss', 'Not set')}")
                if 'take_profit' in best_zone:
                    print(f"   • Take Profit: ${best_zone['take_profit']:.4f}")
        else:
            print(f"\n⏳ Recommendation: WAIT for better setup")
        
        # Profile Analysis
        if context.get('profile_analysis'):
            profile_data = context['profile_analysis']
            print(f"\n👤 {case['profile'].title()} Profile Analysis:")
            print(f"   • Market Phase: {profile_data['market_phase']}")
            print(f"   • Confidence: {profile_data['confidence']:.1%}")
            print(f"   • Entry Signals: {len(profile_data['entry_signals'])}")
            
            if profile_data['warnings']:
                print(f"   • ⚠️  Warnings: {', '.join(profile_data['warnings'])}")
            
            # Show specific entry signals
            for i, signal in enumerate(profile_data['entry_signals'][:2], 1):
                print(f"   • Signal {i}: {signal['type']} ({signal['timeframe']})")
        
        # Quick backtest
        print(f"\n📈 Running Quick Backtest...")
        backtest_results = backtest_smc_strategy(df, context, symbol=case['symbol'])
        
        if 'error' not in backtest_results:
            print(f"   • Total PnL: ${backtest_results['total_pnl']:.2f}")
            print(f"   • Win Rate: {backtest_results['win_rate']:.1%}")
            print(f"   • Total Trades: {backtest_results['total_trades']}")
            print(f"   • Profit Factor: {backtest_results['profit_factor']:.2f}")
            
            # SMC Strategy Breakdown
            breakdown = backtest_results['smc_strategy_breakdown']
            print(f"   • Order Block Trades: {breakdown['order_block_trades']}")
            print(f"   • FVG Trades: {breakdown['fvg_trades']}")
            print(f"   • Zone Trades: {breakdown['zone_trades']}")
        else:
            print(f"   • ❌ Backtest Error: {backtest_results['error']}")

def analyze_single_pair(ticker, symbol, profile='swing'):
    """Analyze a single pair in detail"""
    print(f"🔍 Detailed Analysis: {ticker} ({symbol}) - {profile.title()} Profile")
    print("="*70)
    
    # Get data
    df = fetch_data(ticker, period='1mo', interval='4h')
    
    if df.empty:
        print("❌ No data available")
        return
    
    # SMC Analysis
    context = get_market_context_smc(df, profile=profile, symbol=symbol)
    
    if 'error' in context:
        print(f"❌ Error: {context['error']}")
        return
    
    smc = context['smc_analysis']
    
    # Detailed Structure Analysis
    print(f"\n📊 Market Structure Analysis:")
    structure = smc['structure']
    print(f"   • Structure Type: {structure['structure_type']}")
    print(f"   • Trend Direction: {structure['trend_direction']}")
    print(f"   • Structure Strength: {structure['structure_strength']:.2f}")
    print(f"   • BOS Detected: {structure['bos_detected']}")
    print(f"   • Last Swing High: ${structure['last_swing_high']}")
    print(f"   • Last Swing Low: ${structure['last_swing_low']}")
    
    # Liquidity Analysis
    print(f"\n💧 Liquidity Analysis:")
    print(f"   • Total Liquidity Zones: {len(smc['liquidity_zones'])}")
    print(f"   • Liquidity Sweeps: {len(smc['liquidity_sweeps'])}")
    
    for sweep in smc['liquidity_sweeps'][:3]:  # Show top 3 sweeps
        print(f"     - {sweep['direction']} sweep at ${sweep['swept_price']:.4f} (Strength: {sweep['strength']:.2f})")
    
    # Order Blocks
    print(f"\n📦 Order Block Analysis:")
    active_obs = [ob for ob in smc['order_blocks'] if ob['status'] == 'ACTIVE']
    print(f"   • Active Order Blocks: {len(active_obs)}")
    
    for ob in active_obs[:3]:  # Show top 3 OBs
        print(f"     - {ob['type']}: ${ob['low']:.4f} - ${ob['high']:.4f} (Strength: {ob['strength']:.2f})")
    
    # Fair Value Gaps
    print(f"\n⚡ Fair Value Gap Analysis:")
    active_fvgs = [fvg for fvg in smc['fair_value_gaps'] if fvg['status'] == 'ACTIVE']
    print(f"   • Active FVGs: {len(active_fvgs)}")
    
    for fvg in active_fvgs[:3]:  # Show top 3 FVGs
        print(f"     - {fvg['type']}: ${fvg['bottom']:.4f} - ${fvg['top']:.4f} (Strength: {fvg['strength']:.2f})")
    
    # Premium/Discount Zones
    print(f"\n🎯 Premium/Discount Analysis:")
    zones = smc['premium_discount_zones']
    print(f"   • Current Zone: {zones['current_zone']}")
    print(f"   • Range High: ${zones['range_high']:.4f}")
    print(f"   • Range Low: ${zones['range_low']:.4f}")
    print(f"   • Premium Zone: ${zones['premium']['bottom']:.4f} - ${zones['premium']['top']:.4f}")
    print(f"   • Discount Zone: ${zones['discount']['bottom']:.4f} - ${zones['discount']['top']:.4f}")

if __name__ == "__main__":
    # Run comprehensive test
    test_smc_analysis()
    
    print(f"\n\n{'='*70}")
    print("DETAILED SINGLE PAIR ANALYSIS")
    print(f"{'='*70}")
    
    # Detailed analysis of one pair
    analyze_single_pair('BTC-USD', 'BTCUSD', 'swing')
