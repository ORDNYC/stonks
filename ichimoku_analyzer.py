import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IchimokuAnalyzer:
    def __init__(self, conversion_period=9, base_period=26, span_period=52, displacement=26):
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.span_period = span_period
        self.displacement = displacement
    
    def calculate_ichimoku(self, df):
        """Calculate all Ichimoku components"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['High'].rolling(window=self.conversion_period).max()
        low_9 = df['Low'].rolling(window=self.conversion_period).min()
        df['Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['High'].rolling(window=self.base_period).max()
        low_26 = df['Low'].rolling(window=self.base_period).min()
        df['Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['High'].rolling(window=self.span_period).max()
        low_52 = df['Low'].rolling(window=self.span_period).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span)
        df['Chikou'] = df['Close'].shift(-self.displacement)
        
        # Additional helper columns
        df['Cloud_Top'] = np.maximum(df['Senkou_A'], df['Senkou_B'])
        df['Cloud_Bottom'] = np.minimum(df['Senkou_A'], df['Senkou_B'])
        df['Cloud_Thickness'] = df['Cloud_Top'] - df['Cloud_Bottom']
        df['Cloud_Color'] = np.where(df['Senkou_A'] > df['Senkou_B'], 'green', 'red')
        
        return df
    
    def trend_strength_metrics(self, df, lookback=50):
        """Calculate trend strength indicators"""
        # Cloud Position Score
        price_above_cloud = (df['Close'] > df['Cloud_Top']).rolling(lookback).mean() * 100
        
        # Trend Consistency (cloud color consistency)
        cloud_green = (df['Cloud_Color'] == 'green')
        trend_consistency = cloud_green.rolling(lookback).apply(
            lambda x: max(x.sum(), (lookback - x.sum())) / lookback * 100
        )
        
        # Line Alignment Score (bullish: Tenkan > Kijun > Close > Cloud_top)
        bullish_alignment = (
            (df['Tenkan'] > df['Kijun']) & 
            (df['Kijun'] > df['Close']) & 
            (df['Close'] > df['Cloud_Top'])
        ).rolling(lookback).mean() * 100
        
        bearish_alignment = (
            (df['Tenkan'] < df['Kijun']) & 
            (df['Kijun'] < df['Close']) & 
            (df['Close'] < df['Cloud_Bottom'])
        ).rolling(lookback).mean() * 100
        
        line_alignment_score = np.maximum(bullish_alignment, bearish_alignment)
        
        return {
            'cloud_position_score': price_above_cloud.iloc[-1] if not pd.isna(price_above_cloud.iloc[-1]) else 50,
            'trend_consistency': trend_consistency.iloc[-1] if not pd.isna(trend_consistency.iloc[-1]) else 50,
            'line_alignment_score': line_alignment_score.iloc[-1] if not pd.isna(line_alignment_score.iloc[-1]) else 0
        }
    
    def momentum_quality_metrics(self, df, lookback=50):
        """Calculate momentum quality indicators"""
        # Chikou Clarity - % time Chikou is clearly above/below price
        chikou_diff = df['Chikou'] - df['Close'].shift(self.displacement)
        chikou_clear_signals = (np.abs(chikou_diff) > df['Close'] * 0.02)  # 2% threshold
        chikou_clarity = chikou_clear_signals.rolling(lookback).mean() * 100
        
        # Tenkan/Kijun crosses for momentum persistence
        tk_cross = ((df['Tenkan'] > df['Kijun']) != (df['Tenkan'].shift(1) > df['Kijun'].shift(1)))
        cross_frequency = tk_cross.rolling(lookback).sum()
        
        return {
            'chikou_clarity': chikou_clarity.iloc[-1] if not pd.isna(chikou_clarity.iloc[-1]) else 50,
            'cross_frequency': cross_frequency.iloc[-1] if not pd.isna(cross_frequency.iloc[-1]) else 0,
            'momentum_persistence': 100 - (cross_frequency.iloc[-1] * 2) if not pd.isna(cross_frequency.iloc[-1]) else 50
        }
    
    def volatility_metrics(self, df, lookback=50):
        """Calculate volatility and support/resistance metrics"""
        # Cloud Thickness Ratio
        cloud_thickness_ratio = (df['Cloud_Thickness'] / df['Close']).rolling(lookback).mean() * 100
        
        # Cloud Volatility
        cloud_volatility = (df['Cloud_Thickness'] / df['Close']).rolling(lookback).std() * 100
        
        # Support/Resistance effectiveness
        price_at_cloud = (
            (df['Close'] <= df['Cloud_Top']) & 
            (df['Close'] >= df['Cloud_Bottom'])
        )
        sr_tests = price_at_cloud.rolling(5).sum() > 0  # Price tested cloud in last 5 days
        
        return {
            'cloud_thickness_ratio': cloud_thickness_ratio.iloc[-1] if not pd.isna(cloud_thickness_ratio.iloc[-1]) else 0,
            'cloud_volatility': cloud_volatility.iloc[-1] if not pd.isna(cloud_volatility.iloc[-1]) else 0,
            'sr_effectiveness': sr_tests.rolling(lookback).mean().iloc[-1] * 100 if not pd.isna(sr_tests.rolling(lookback).mean().iloc[-1]) else 0
        }
    
    def signal_quality_metrics(self, df, lookback=50):
        """Calculate signal quality indicators"""
        # Tenkan/Kijun cross analysis
        tk_bullish_cross = ((df['Tenkan'] > df['Kijun']) & (df['Tenkan'].shift(1) <= df['Kijun'].shift(1)))
        tk_bearish_cross = ((df['Tenkan'] < df['Kijun']) & (df['Tenkan'].shift(1) >= df['Kijun'].shift(1)))
        
        # Signal follow-through (5-day return after cross)
        bullish_returns = []
        bearish_returns = []
        
        for i in range(5, len(df)-5):
            if tk_bullish_cross.iloc[i]:
                ret = (df['Close'].iloc[i+5] - df['Close'].iloc[i]) / df['Close'].iloc[i] * 100
                bullish_returns.append(ret)
            elif tk_bearish_cross.iloc[i]:
                ret = (df['Close'].iloc[i] - df['Close'].iloc[i+5]) / df['Close'].iloc[i] * 100
                bearish_returns.append(ret)
        
        avg_signal_return = np.mean(bullish_returns + bearish_returns) if (bullish_returns + bearish_returns) else 0
        signal_success_rate = np.mean([r > 0 for r in (bullish_returns + bearish_returns)]) * 100 if (bullish_returns + bearish_returns) else 50
        
        return {
            'avg_signal_return': avg_signal_return,
            'signal_success_rate': signal_success_rate
        }
    
    def confluence_score(self, df):
        """Calculate current Ichimoku confluence score (0-5)"""
        latest = df.iloc[-1]
        score = 0
        
        # 1. Price above cloud
        if latest['Close'] > latest['Cloud_Top']:
            score += 1
        elif latest['Close'] < latest['Cloud_Bottom']:
            score += 1
            
        # 2. Tenkan above Kijun (or below for bearish)
        if ((latest['Close'] > latest['Cloud_Top']) and (latest['Tenkan'] > latest['Kijun'])) or \
           ((latest['Close'] < latest['Cloud_Bottom']) and (latest['Tenkan'] < latest['Kijun'])):
            score += 1
            
        # 3. Green cloud (or red for bearish trend)
        if ((latest['Close'] > latest['Cloud_Top']) and (latest['Cloud_Color'] == 'green')) or \
           ((latest['Close'] < latest['Cloud_Bottom']) and (latest['Cloud_Color'] == 'red')):
            score += 1
            
        # 4. Chikou above price (or below for bearish)
        chikou_val = df['Chikou'].shift(self.displacement).iloc[-1]
        if not pd.isna(chikou_val):
            if ((latest['Close'] > latest['Cloud_Top']) and (chikou_val > df['Close'].shift(self.displacement).iloc[-1])) or \
               ((latest['Close'] < latest['Cloud_Bottom']) and (chikou_val < df['Close'].shift(self.displacement).iloc[-1])):
                score += 1
        
        # 5. Strong trend (price well away from cloud)
        if latest['Close'] > latest['Cloud_Top'] * 1.02 or latest['Close'] < latest['Cloud_Bottom'] * 0.98:
            score += 1
            
        return score
    
    def analyze_stock(self, symbol, period='1y'):
        """Complete analysis for a single stock"""
        try:
            # Download data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                return None
                
            # Calculate Ichimoku
            df = self.calculate_ichimoku(df)
            
            # Calculate all metrics
            trend_metrics = self.trend_strength_metrics(df)
            momentum_metrics = self.momentum_quality_metrics(df)
            volatility_metrics = self.volatility_metrics(df)
            signal_metrics = self.signal_quality_metrics(df)
            confluence = self.confluence_score(df)
            
            # Combine all metrics
            analysis = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'data_points': len(df),
                
                # Trend Strength
                'cloud_position_score': round(trend_metrics['cloud_position_score'], 2),
                'trend_consistency': round(trend_metrics['trend_consistency'], 2),
                'line_alignment_score': round(trend_metrics['line_alignment_score'], 2),
                
                # Momentum Quality
                'chikou_clarity': round(momentum_metrics['chikou_clarity'], 2),
                'cross_frequency': int(momentum_metrics['cross_frequency']),
                'momentum_persistence': round(momentum_metrics['momentum_persistence'], 2),
                
                # Volatility
                'cloud_thickness_ratio': round(volatility_metrics['cloud_thickness_ratio'], 4),
                'cloud_volatility': round(volatility_metrics['cloud_volatility'], 4),
                'sr_effectiveness': round(volatility_metrics['sr_effectiveness'], 2),
                
                # Signal Quality
                'avg_signal_return': round(signal_metrics['avg_signal_return'], 2),
                'signal_success_rate': round(signal_metrics['signal_success_rate'], 2),
                
                # Overall
                'confluence_score': confluence,
                
                # Current values
                'current_price': round(df['Close'].iloc[-1], 2),
                'current_trend': 'Bullish' if df['Close'].iloc[-1] > df['Cloud_Top'].iloc[-1] else 
                               'Bearish' if df['Close'].iloc[-1] < df['Cloud_Bottom'].iloc[-1] else 'Neutral'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def compare_stocks(self, symbols, period='1y'):
        """Analyze multiple stocks and return comparison dataframe"""
        results = []
        
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            analysis = self.analyze_stock(symbol, period)
            if analysis:
                results.append(analysis)
        
        if not results:
            return None
            
        df_results = pd.DataFrame(results)
        
        # Calculate composite scores
        df_results['trend_strength_composite'] = (
            df_results['cloud_position_score'] * 0.4 +
            df_results['trend_consistency'] * 0.3 +
            df_results['line_alignment_score'] * 0.3
        ).round(2)
        
        df_results['momentum_quality_composite'] = (
            df_results['chikou_clarity'] * 0.4 +
            df_results['momentum_persistence'] * 0.6
        ).round(2)
        
        df_results['overall_ichimoku_score'] = (
            df_results['trend_strength_composite'] * 0.3 +
            df_results['momentum_quality_composite'] * 0.25 +
            df_results['signal_success_rate'] * 0.25 +
            df_results['confluence_score'] * 20 * 0.2  # Scale confluence to 0-100
        ).round(2)
        
        return df_results.sort_values('overall_ichimoku_score', ascending=False)

# Testing and Usage Example
def test_ichimoku_analyzer():
    """Test the analyzer with sample stocks"""
    analyzer = IchimokuAnalyzer()
    
    # Test with popular stocks
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print("Starting Ichimoku Analysis...")
    print("=" * 50)
    
    results = analyzer.compare_stocks(test_symbols, period='6mo')
    
    if results is not None:
        print("\nRESULTS SUMMARY:")
        print("=" * 50)
        
        # Display key metrics
        display_cols = [
            'symbol', 'current_trend', 'overall_ichimoku_score',
            'trend_strength_composite', 'momentum_quality_composite',
            'confluence_score', 'signal_success_rate'
        ]
        
        print(results[display_cols].to_string(index=False))
        
        print("\nDETAILED METRICS:")
        print("=" * 50)
        
        for _, row in results.iterrows():
            print(f"\n{row['symbol']} - {row['current_trend']} Trend:")
            print(f"  Overall Score: {row['overall_ichimoku_score']}")
            print(f"  Cloud Position: {row['cloud_position_score']}%")
            print(f"  Trend Consistency: {row['trend_consistency']}%")
            print(f"  Chikou Clarity: {row['chikou_clarity']}%")
            print(f"  Signal Success: {row['signal_success_rate']}%")
            print(f"  Confluence Score: {row['confluence_score']}/5")
            print(f"  Cross Frequency: {row['cross_frequency']} (lower is better)")
        
        return results
    else:
        print("No results obtained. Check symbols and internet connection.")
        return None

if __name__ == "__main__":
    # Run the test
    results = test_ichimoku_analyzer()
