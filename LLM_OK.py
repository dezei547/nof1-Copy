import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import re
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
import numpy as np

import math
load_dotenv()

# 初始化AI客户端 - 使用DeepSeek Reasoning模型
ai_clients = {
    'deepseek': OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
}

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',
        'sandbox': True,
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),
    'sandbox': True,
})

# 多币种交易配置
TRADE_CONFIG = {
    'symbols': {
        'BTC': 'BTC-USDT-SWAP',
        'ETH': 'ETH-USDT-SWAP', 
        'SOL': 'SOL-USDT-SWAP',
        'BNB': 'BNB-USDT-SWAP',
        'XRP': 'XRP-USDT-SWAP',
        'DOGE': 'DOGE-USDT-SWAP'
    },
    'leverage': 20,
    'timeframe': '3m',
    'test_mode': False,
    'data_points': 96,
    'ai_provider': 'deepseek',
    'position_management': {
        'base_usdt_amount': 100,
        'high_confidence_multiplier': 1.5,
        'medium_confidence_multiplier': 1.0,
        'low_confidence_multiplier': 0.5,
        'max_position_ratio': 10,
        'max_concurrent_positions': 8  # 最大同时持仓数量
    }
}

# 全局变量
price_history = {}
signal_history = []
positions = {}
start_time = datetime.now()
execution_count = 0
balance_history = []  # 新增：用于存储账户余额历史
# 创建存储日志的目录
LOG_DIR = "deepseek_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def calculate_sharpe_ratio():
    """计算夏普比率
    基于周期收益率计算，不进行年化，直接返回周期级别的夏普比率
    公式：夏普比率 = 平均收益率 / 收益率标准差
    """
    try:
        # 使用全局的balance_history
        global balance_history
        
        if len(balance_history) < 2:
            return 0.0  # 数据不足时返回0
        
        # 提取每个周期的账户净值
        equities = [bh['balance'] for bh in balance_history if bh['balance'] > 0]
        
        if len(equities) < 2:
            return 0.0
        
        # 计算周期收益率
        returns = []
        for i in range(1, len(equities)):
            prev_equity = equities[i-1]
            curr_equity = equities[i]
            period_return = (curr_equity - prev_equity) / prev_equity
            returns.append(period_return)
        
        if len(returns) == 0:
            return 0.0
        
        # 计算平均收益率
        mean_return = sum(returns) / len(returns)
        
        # 计算收益率标准差
        sum_squared_diff = 0.0
        for r in returns:
            diff = r - mean_return
            sum_squared_diff += diff * diff
        variance = sum_squared_diff / len(returns)
        std_dev = math.sqrt(variance)
        
        # 避免除以零
        if std_dev == 0:
            if mean_return > 0:
                return 999.0  # 无波动的正收益
            elif mean_return < 0:
                return -999.0  # 无波动的负收益
            return 0.0
        
        # 计算夏普比率（不进行年化）
        sharpe_ratio = mean_return / std_dev
        
        # 打印计算过程
        print(f"📊 夏普比率计算: {len(returns)}个周期数据点")
        print(f"  平均周期收益率: {mean_return:.4%}")
        print(f"  收益率标准差: {std_dev:.4%}")
        print(f"  周期夏普比率: {sharpe_ratio:.3f}")
        
        return sharpe_ratio
        
    except Exception as e:
        print(f"计算夏普比率失败: {e}")
        return 0.0

def get_ohlcv_data(symbol, timeframe='3m', limit=100):
    """获取K线数据"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"获取{symbol} K线数据失败: {e}")
        return None

def calculate_technical_indicators(df):
    """计算技术指标 - 使用pandas内置方法"""
    try:
        if df is None or len(df) < 50:
            return None
            
        # 使用pandas计算EMA
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        
        # 计算MACD (12,26,9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        # 计算RSI
        def calculate_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        rsi_7 = calculate_rsi(df['close'], 7)
        rsi_14 = calculate_rsi(df['close'], 14)
        
        # 计算ATR (真实波动幅度)
        def calculate_atr(high, low, close, period):
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
            
        atr_3 = calculate_atr(df['high'], df['low'], df['close'], 3)
        atr_14 = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        return {
            'ema_20': ema_20.values,
            'ema_50': ema_50.values,
            'macd': macd.values,
            'macd_signal': macd_signal.values,
            'macd_hist': macd_hist.values,
            'rsi_7': rsi_7.values,
            'rsi_14': rsi_14.values,
            'atr_3': atr_3.values,
            'atr_14': atr_14.values
        }
    except Exception as e:
        print(f"计算技术指标失败: {e}")
        return None

def get_funding_rate(symbol):
    """获取资金费率"""
    try:
        # OKX获取资金费率
        funding_data = exchange.public_get_public_funding_rate({'instId': symbol})
        if funding_data and 'data' in funding_data and len(funding_data['data']) > 0:
            return float(funding_data['data'][0]['fundingRate'])
        return 0.0
    except Exception as e:
        print(f"获取{symbol}资金费率失败: {e}")
        return 0.0

def get_open_interest(symbol):
    """获取未平仓合约量"""
    try:
        # OKX获取未平仓合约
        oi_data = exchange.public_get_public_open_interest({'instId': symbol})
        if oi_data and 'data' in oi_data and len(oi_data['data']) > 0:
            return float(oi_data['data'][0]['oi'])
        return 0.0
    except Exception as e:
        print(f"获取{symbol}未平仓合约失败: {e}")
        return 0.0

def get_avg_open_interest(symbol, days=7):
    """获取平均未平仓合约量"""
    try:
        # 这里简化处理，实际应该获取历史数据计算平均值
        current_oi = get_open_interest(symbol)
        return current_oi * 0.95  # 模拟平均值
    except Exception as e:
        print(f"获取{symbol}平均未平仓合约失败: {e}")
        return get_open_interest(symbol)

def get_current_price(symbol):
    """获取当前价格"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"获取{symbol}当前价格失败: {e}")
        return 0.0

def get_intraday_series_data(symbol, timeframe='3m', limit=96):
    """获取日内序列数据"""
    try:
        df = get_ohlcv_data(symbol, timeframe, limit)
        if df is None or len(df) < 20:
            return None
            
        indicators = calculate_technical_indicators(df)
        if indicators is None:
            return None
            
        # 构建日内序列
        intraday_data = {
            'mid_price': df['close'].tolist(),  # 使用收盘价作为中间价
            'ema_20': indicators['ema_20'].tolist() if indicators['ema_20'] is not None else [],
            'macd': indicators['macd'].tolist() if indicators['macd'] is not None else [],
            'rsi_7': indicators['rsi_7'].tolist() if indicators['rsi_7'] is not None else [],
            'rsi_14': indicators['rsi_14'].tolist() if indicators['rsi_14'] is not None else []
        }
        
        return intraday_data
    except Exception as e:
        print(f"获取{symbol}日内序列数据失败: {e}")
        return None

def get_long_term_context(symbol):
    """获取长期背景数据（4小时级别）"""
    try:
        # 获取4小时K线数据
        df_4h = get_ohlcv_data(symbol, '4h', limit=50)
        if df_4h is None or len(df_4h) < 50:
            return {}
            
        indicators_4h = calculate_technical_indicators(df_4h)
        if indicators_4h is None:
            return {}
            
        # 计算成交量指标
        current_volume = df_4h['volume'].iloc[-1] if len(df_4h) > 0 else 0
        avg_volume = df_4h['volume'].tail(20).mean() if len(df_4h) >= 20 else current_volume
        
        return {
            'ema_20_4h': indicators_4h['ema_20'][-1] if indicators_4h['ema_20'] is not None else 0,
            'ema_50_4h': indicators_4h['ema_50'][-1] if indicators_4h['ema_50'] is not None else 0,
            'atr_3_4h': indicators_4h['atr_3'][-1] if indicators_4h['atr_3'] is not None else 0,
            'atr_14_4h': indicators_4h['atr_14'][-1] if indicators_4h['atr_14'] is not None else 0,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'macd_4h': indicators_4h['macd'].tolist() if indicators_4h['macd'] is not None else [],
            'rsi_14_4h': indicators_4h['rsi_14'].tolist() if indicators_4h['rsi_14'] is not None else []
        }
    except Exception as e:
        print(f"获取{symbol}长期背景数据失败: {e}")
        return {}

def get_single_crypto_data(symbol_name):
    """获取单个币种的完整数据"""
    try:
        symbol = TRADE_CONFIG['symbols'][symbol_name]
        
        # 获取当前价格
        current_price = get_current_price(symbol)
        
        # 获取日内序列数据
        intraday_series = get_intraday_series_data(symbol, TRADE_CONFIG['timeframe'], TRADE_CONFIG['data_points'])
        if intraday_series is None:
            return None
            
        # 获取技术指标
        df = get_ohlcv_data(symbol, TRADE_CONFIG['timeframe'], TRADE_CONFIG['data_points'])
        indicators = calculate_technical_indicators(df)
        
        # 获取资金费率和未平仓合约
        funding_rate = get_funding_rate(symbol)
        open_interest = get_open_interest(symbol)
        avg_open_interest = get_avg_open_interest(symbol)
        
        # 获取长期背景数据
        long_term_context = get_long_term_context(symbol)
        
        return {
            'symbol': symbol_name,
            'current_price': current_price,
            'current_ema20': indicators['ema_20'][-1] if indicators and indicators['ema_20'] is not None else current_price,
            'current_macd': indicators['macd'][-1] if indicators and indicators['macd'] is not None else 0,
            'current_rsi_7': indicators['rsi_7'][-1] if indicators and indicators['rsi_7'] is not None else 50,
            'current_rsi_14': indicators['rsi_14'][-1] if indicators and indicators['rsi_14'] is not None else 50,
            'funding_rate': funding_rate,
            'open_interest': open_interest,
            'avg_open_interest': avg_open_interest,
            'intraday_series': intraday_series,
            'long_term_context': long_term_context
        }
    except Exception as e:
        print(f"获取{symbol_name}完整数据失败: {e}")
        return None

def get_all_crypto_data():
    """获取所有币种的完整数据"""
    crypto_data = {}
    
    print("📈 开始获取全市场数据...")
    for symbol_name in TRADE_CONFIG['symbols'].keys():
        print(f"  正在获取 {symbol_name} 数据...")
        data = get_single_crypto_data(symbol_name)
        if data:
            crypto_data[symbol_name] = data
            print(f"  ✅ {symbol_name} 数据获取成功")
        else:
            print(f"  ❌ {symbol_name} 数据获取失败")
        
        # 避免API限制
        time.sleep(0.5)
    
    print(f"✅ 全市场数据获取完成，共获取{len(crypto_data)}个币种数据")
    return crypto_data

def get_account_balance():
    """获取账户余额"""
    try:
        balance = exchange.fetch_balance()
        total_balance = balance['total']['USDT'] if 'USDT' in balance['total'] else 0
        free_balance = balance['free']['USDT'] if 'USDT' in balance['free'] else 0
        return total_balance, free_balance
    except Exception as e:
        print(f"获取账户余额失败: {e}")
        return 0, 0
def get_current_position_for_symbol(symbol):
    """获取指定币种的当前持仓 - 支持符号转换"""
    try:
        # 获取配置中的符号
        config_symbol = TRADE_CONFIG['symbols'][symbol]  # 如: BNB-USDT-SWAP
        
        # 转换为交易所持仓查询格式
        exchange_symbol = config_symbol.replace('-USDT-SWAP', '/USDT:USDT').replace('-', '/')
        
        print(f"🔍 查询{symbol}持仓: 配置符号={config_symbol}, 交易所符号={exchange_symbol}")
        
        positions = exchange.fetch_positions([exchange_symbol])
        
        for pos in positions:
            # 检查符号匹配（支持两种格式）
            if (pos['symbol'] == exchange_symbol or 
                pos['symbol'] == config_symbol) and pos['contracts']:
                
                contracts = float(pos['contracts'])
                if contracts > 0:
                    position_info = {
                        'side': pos['side'],
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'leverage': pos.get('leverage', TRADE_CONFIG['leverage']),
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'liquidation_price': float(pos['liquidationPrice']) if pos['liquidationPrice'] else 0
                    }
                    print(f"✅ 找到{symbol}持仓: {position_info['side']} {contracts}张")
                    return position_info
        
        print(f"💤 {symbol}无持仓")
        return None
        
    except Exception as e:
        print(f"❌ 获取{symbol}持仓失败: {e}")
        return None
def calculate_stop_loss_take_profit(side, entry_price, current_price, symbol):
    """计算止损止盈价格"""
    try:



        stop_loss_percent = 0.015  # 5%
        take_profit_percent = 0.03  # 10%

        if side == 'long':
            stop_loss = entry_price * (1 - stop_loss_percent)
            take_profit = entry_price * (1 + take_profit_percent)
        elif side == 'short':
            stop_loss = entry_price * (1 + stop_loss_percent)
            take_profit = entry_price * (1 - take_profit_percent)
        else:
            # 默认值
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.04
        print(f"🔍 {symbol} 计算结果 - 止损: {stop_loss:.3f}, 止盈: {take_profit:.3f}")
        return stop_loss, take_profit
        
    except Exception as e:
        print(f"计算止损止盈失败: {e}")
        # 返回默认值
        if side == 'long':
            return entry_price * 0.98, entry_price * 1.04
        else:
            return entry_price * 1.02, entry_price * 0.96

def calculate_current_atr(symbol):
    """计算当前ATR值 - 修复版本"""
    try:
        # 检查传入的symbol是币种名称还是完整符号
        if symbol in TRADE_CONFIG['symbols']:
            full_symbol = TRADE_CONFIG['symbols'][symbol]
            symbol_name = symbol
        else:
            full_symbol = symbol
            symbol_name = symbol.split('-')[0]
        
        print(f"🔍 计算{symbol_name}的ATR，使用符号: {full_symbol}")
            
        # 获取更多数据点来处理滚动计算
        df = get_ohlcv_data(full_symbol, TRADE_CONFIG['timeframe'], 20)
        if df is None or len(df) < 15:  # 需要至少15个点来计算14期ATR
            print(f"❌ 获取{symbol_name} K线数据失败或数据不足")
            return 0
        
        # 确保数据是数值类型
        df = df.astype({
            'high': float, 
            'low': float, 
            'close': float
        })
        
        # 方法1：使用手动计算，避免pandas Series问题
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        true_ranges = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]  # 当日高低点差
            tr2 = abs(high[i] - close[i-1])  # 当日高点-前日收盘
            tr3 = abs(low[i] - close[i-1])   # 当日低点-前日收盘
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # 计算14期ATR
        if len(true_ranges) >= 14:
            atr_value = sum(true_ranges[-14:]) / 14
        else:
            atr_value = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        
        # 确保返回float类型
        atr_value = float(atr_value)
        
        print(f"✅ {symbol_name} ATR计算成功: {atr_value:.4f}")
        print(f"📊 数据详情: 价格范围 {high[-1]:.2f}-{low[-1]:.2f}, 最近TR: {true_ranges[-1]:.4f}")
        
        return atr_value
        
    except Exception as e:
        print(f"❌ 计算{symbol} ATR失败: {e}")
        import traceback
        traceback.print_exc()
        return 0

def get_current_positions():
    """获取当前持仓 - 修复版本"""
    print("🔍 获取当前持仓...")
    try:
        positions = exchange.fetch_positions()
        current_positions = []
        
        print(f"📊 原始持仓数据条数: {len(positions)}")
        
        # 创建符号转换映射
        def convert_symbol_to_config(original_symbol):
            """将持仓符号转换为配置符号格式"""
            if original_symbol.endswith('/USDT:USDT'):
                base_currency = original_symbol.split('/')[0]
                return f"{base_currency}-USDT-SWAP"
            return original_symbol
        
        for pos in positions:
            original_symbol = pos.get('symbol', '')
            config_symbol = convert_symbol_to_config(original_symbol)
            
            contracts = pos.get('contracts', 0)
            
            if contracts and float(contracts) > 0:
                try:
                    entry_price = float(pos['entryPrice']) if pos['entryPrice'] else 0
                    current_price = get_current_price(original_symbol)
                    unrealized_pnl = float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0
                    
                    # 计算止损止盈价格
                    stop_loss, take_profit = calculate_stop_loss_take_profit(
                        pos.get('side', 'unknown'), 
                        entry_price, 
                        current_price,
                        config_symbol
                    )
                    
                    position_info = {
                        'symbol': config_symbol,
                        'original_symbol': original_symbol,
                        'quantity': float(contracts),
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'liquidation_price': float(pos['liquidationPrice']) if pos['liquidationPrice'] else 0,
                        'unrealized_pnl': unrealized_pnl,
                        'leverage': pos.get('leverage', TRADE_CONFIG['leverage']),
                        'side': pos.get('side', 'unknown'),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    current_positions.append(position_info)
                    print(f"  ✅ 发现有效持仓: {config_symbol} {position_info['side']} {contracts}张")
                    print(f"  📊 止损: {stop_loss:.3f}, 止盈: {take_profit:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ 解析持仓数据失败: {e}")
        
        print(f"📊 最终有效持仓: {len(current_positions)}个")
        return current_positions
        
    except Exception as e:
        print(f"获取持仓失败: {e}")
        return []

def calculate_total_return(initial_balance=1000):
    """计算总回报率"""
    try:
        total_balance, _ = get_account_balance()
        total_return = ((total_balance - initial_balance) / initial_balance) * 100
        return total_return
    except Exception as e:
        print(f"计算总回报率失败: {e}")
        return 0.0


def get_account_performance():
    """获取账户绩效数据"""
    print("💰 获取账户绩效数据...")
    try:
        total_balance, free_balance = get_account_balance()
        current_positions = get_current_positions()
        total_return = calculate_total_return()
        sharpe_ratio = calculate_sharpe_ratio()
        
        # 计算持仓总价值
        positions_value = sum([pos['quantity'] * pos['current_price'] for pos in current_positions])
        
        return {
            'total_return_percent': total_return,
            'available_cash': free_balance,
            'account_value': total_balance,
            'positions': current_positions,
            'total_positions': len(current_positions),
            'sharpe_ratio': sharpe_ratio,
            'positions_value': positions_value
        }
    except Exception as e:
        print(f"获取账户绩效失败: {e}")
        # 返回默认值
        return {
            'total_return_percent': 0.0,
            'available_cash': 1000,
            'account_value': 1000,
            'positions': [],
            'total_positions': 0,
            'sharpe_ratio': 1.0,
            'positions_value': 0
        }

# 原有的其他函数保持不变（setup_exchange, create_deepseek_reasoning_prompt等）
def setup_exchange():
    """设置交易所参数 - 多币种版本"""
    try:
        print("🔍 初始化交易所连接...")
        
        # 测试API连接
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
        print(f"✅ API连接成功，USDT余额: {usdt_balance:.2f}")
        
        # 加载市场数据
        markets = exchange.load_markets()
        print(f"✅ 加载市场数据成功，共{len(markets)}个交易对")
        
        # 为每个交易对设置杠杆
        for symbol_name, symbol in TRADE_CONFIG['symbols'].items():
            if symbol in markets:
                market = markets[symbol]
                contract_size = market.get('contractSize', '未知')
                print(f"✅ {symbol_name}合约规格: 1张 = {contract_size} {symbol_name}")
                
                # 设置杠杆
                try:
                    exchange.set_leverage(
                        TRADE_CONFIG['leverage'],
                        symbol,
                        params={'mgnMode': 'cross'}
                    )
                    print(f"✅ {symbol_name} - 已设置全仓模式，杠杆倍数: {TRADE_CONFIG['leverage']}x")
                except Exception as e:
                    print(f"⚠️ {symbol_name}设置杠杆失败: {e}")

        # 检查账户模式
        check_account_mode()
        
        return True

    except Exception as e:
        print(f"❌ 交易所设置失败: {e}")
        return False

def create_deepseek_reasoning_prompt(crypto_data, account_info):
    """创建DeepSeek Reasoning模型的提示词模板 - 多币种版本"""
    global start_time, execution_count
    execution_count += 1
    elapsed_minutes = int((datetime.now() - start_time).total_seconds() / 60)
        # 获取最新夏普比率（新增）
    latest_sharpe = calculate_sharpe_ratio()
    prompt = f"""你是一位专业的加密货币日内交易员，专注于3分钟级别的短期交易机会。。请基于以下数据和逻辑进行严谨的分析。自您开始交易以来已经过去了 {elapsed_minutes} 分钟。当前时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}，您已被调用 {execution_count} 次。下面，我们为您提供各种状态数据、价格数据和预测信号，以便您发现阿尔法。下面是您的往来账户信息、价值、业绩、头寸等。

以下所有价格或信号数据均按顺序排列：最旧→最新

时间范围说明：除非章节标题中另有说明，否则日内系列以 3 分钟的间隔提供。如果代币使用不同的区间，则在该代币的部分中明确说明。

所有代币的当前市场状况
"""

    # 添加每个币种的详细数据
    for symbol_name in TRADE_CONFIG['symbols'].keys():
        if symbol_name in crypto_data:
            data = crypto_data[symbol_name]
            
            # 构建日内序列数据
            mid_prices = [f"{p:.3f}" for p in data['intraday_series']['mid_price'][-10:]]
            ema_20 = [f"{e:.3f}" for e in data['intraday_series']['ema_20'][-10:]]
            macd = [f"{m:.3f}" for m in data['intraday_series']['macd'][-10:]]
            rsi_7 = [f"{r:.3f}" for r in data['intraday_series']['rsi_7'][-10:]]
            rsi_14 = [f"{r:.3f}" for r in data['intraday_series']['rsi_14'][-10:]]
            
            # 构建长期背景数据
            long_term = data['long_term_context']
            macd_4h = [f"{m:.3f}" for m in long_term.get('macd_4h', [])[-10:]] if long_term.get('macd_4h') else []
            rsi_14_4h = [f"{r:.3f}" for r in long_term.get('rsi_14_4h', [])[-10:]] if long_term.get('rsi_14_4h') else []
            
            prompt += f"""所有 {symbol_name} 数据
当前价格 = {data['current_price']:.3f}, 当前EMA20 = {data['current_ema20']:.3f}, 当前MACD = {data['current_macd']:.3f}, 当前RSI (7周期) = {data['current_rsi_7']:.3f}

此外，以下是 perps（您正在交易的工具）的最新 {symbol_name} 未平仓合约和资金费率：

未平仓合约：最新：{data['open_interest']:.2f} 平均：{data.get('avg_open_interest', data['open_interest']):.2f}

资金费率：{data['funding_rate']:.6f}

日内系列（按分钟，最旧→最新）：

中间价：{mid_prices}

EMA指标（20周期）：{ema_20}

MACD指标：{macd}

RSI指标（7周期）：{rsi_7}

RSI指标（14周期）：{rsi_14}

长期背景（4小时时间范围）：

20周期EMA：{long_term.get('ema_20_4h', 0):.3f} 对比 50周期EMA：{long_term.get('ema_50_4h', 0):.3f}

3周期ATR：{long_term.get('atr_3_4h', 0):.3f} 对比 14周期ATR：{long_term.get('atr_14_4h', 0):.3f}

当前成交量：{long_term.get('current_volume', 0):.3f} 对比 平均成交量：{long_term.get('avg_volume', 0):.3f}

MACD指标：{macd_4h}

RSI指标（14周期）：{rsi_14_4h}

"""

    prompt += f"""这是您的账户信息和表现
当前总回报率（百分比）：{account_info['total_return_percent']:.2f}%

可用现金：{account_info['available_cash']:.1f}

当前账户价值：{account_info['account_value']:.2f}

当前现场头寸和表现："""

    if account_info['positions']:
        for pos in account_info['positions']:
            prompt += f"""
    {pos['symbol']}持仓详情：
    - 方向: {pos['side']}
    - 数量: {pos['quantity']:.2f}张
    - 入场价格: {pos['entry_price']:.3f}
    - 当前价格: {pos['current_price']:.3f}
    - 当前盈亏: {pos['unrealized_pnl']:.2f} USDT
    - 杠杆: {pos.get('leverage', 1)}x
    - 强平价格: {pos.get('liquidation_price', 0):.3f}
    - 建议止损: {pos.get('stop_loss', 0):.3f}
    - 建议止盈: {pos.get('take_profit', 0):.3f}
    """
    else:
        prompt += " 无持仓"

    prompt += f"""

夏普比率：最新：{latest_sharpe:.3f}

强烈建议按照以下分析要求，生成你的分析然后构建json数组：
【分析要求】
1. **检查现有头寸状态**
   - 查看每个持仓的盈亏情况
   - 对照退出计划
        检查是否触发止损、止盈或失效条件
   - 评估技术指标（RSI、MACD、EMA、VWAP等）
    除非触发止损和失效条件，否则不要SELL或者CLOSE持仓。尽量长线持有高置信度头寸。技术指标只是暂时的，趋势是长线的。
   
2. **评估新交易机会**,非常重要
   - 检查可用现金和仓位规模
   - 分析市场数据的阿尔法信号
   - 确认是否有强烈的入场信号 ，否则不要生成BUY信号。不要频繁交易，尽量长线持有高置信度头寸

3. **制定交易决策**
   - 对于现有持仓：决定持有还是关闭
   - 对于新交易：决定是否入场
   - 所有决策必须基于数据和退出计划

对于非持仓币种，基于技术指标和价格位置生成交易信号（BUY/SELL/HOLD/CLOSE）。只有在信心为HIGH的情况下，才能生成BUY、SELL信号。如果达到最大持仓限制，则只能发出HOLD信号。
【构建json格式】
请为每个币种构建独立的交易决策，使用以下JSON格式：

[
  {{
    "symbol": "BTC",
    "signal": "BUY/SELL/HOLD/CLOSE",
    "confidence": "HIGH/MEDIUM/LOW", 
    "profit_target": 止盈价格,
    "reason": "详细的交易理由",
    "position_size": "仓位大小描述",
    "stop_loss": 止损价格,
    "invalidation_condition": "无效条件描述",例如；If the price closes below 4000 on a 3-minute candle
    "take_profit": 止盈价格,
    "leverage": 杠杆倍数,
    "risk_usd": 619.23
  }},
  ...
]

严格按照分析要求分析，然后再构建json，
"""

    return prompt

def analyze_with_deepseek_reasoning():
    """使用DeepSeek Reasoning模型进行分析 - 多币种版本"""
    try:
        # 获取所有加密货币数据
        print("📈 获取全市场数据...")
        crypto_data = get_all_crypto_data()
        
        # 获取账户绩效
        print("💰 获取账户绩效...")
        account_info = get_account_performance()
        
        # 创建推理提示词
        prompt = create_deepseek_reasoning_prompt(crypto_data, account_info)
        print("🤖 调用DeepSeek Reasoning模型进行分析...")
        # 调用DeepSeek Reasoning模型
        ai_client = ai_clients['deepseek']
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的量化交易员，擅长多币种组合管理和风险控制。请基于数据和逻辑进行严谨的分析。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1,
            max_tokens=None
        )
        
        result = response.choices[0].message.content
        print(f"DeepSeek Reasoning原始回复: {result}")
        
        # 保存prompt、response和result
        save_deepseek_interaction(prompt, response, result)
        
        # 解析JSON响应
        try:
            # 查找JSON数组
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = result[start_idx:end_idx]
                signals = json.loads(json_str)
            else:
                signals = create_fallback_signals()
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            signals = create_fallback_signals()
            
        return signals
        
    except Exception as e:
        print(f"DeepSeek Reasoning分析失败: {e}")
        return create_fallback_signals()

def save_deepseek_interaction(prompt, response, result):
    """保存DeepSeek的完整交互内容（prompt、response和result）"""
    global execution_count
    try:
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{LOG_DIR}/deepseek_interaction_{timestamp}_exec_{execution_count}.txt"
        
        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"===== DeepSeek 交互记录 - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            f.write(f"===== 执行次数: {execution_count} =====\n\n")
            
            f.write("===== PROMPT (提示词) =====\n")
            f.write(prompt)
            f.write("\n\n===== RAW RESPONSE (原始响应) =====\n")
            f.write(str(response))  # 保存完整的response对象
            f.write("\n\n===== PARSED RESULT (解析结果) =====\n")
            f.write(result)
        
        print(f"📄 DeepSeek完整交互已保存到: {filename}")
        return True
    except Exception as e:
        print(f"❌ 保存DeepSeek交互失败: {e}")
        return False

def create_fallback_signals():
    """创建备用信号 - 多币种版本"""
    fallback_signals = []
    for symbol_name in TRADE_CONFIG['symbols'].keys():
        fallback_signals.append({
            "symbol": symbol_name,
            "signal": "HOLD",
            "confidence": "LOW",
            "reason": "系统暂时不可用，采取保守策略",
            "position_size": "无",
            "stop_loss": 0,
            "take_profit": 0,
            "leverage": TRADE_CONFIG['leverage']
        })
    return fallback_signals


def execute_trades(signals):
    """执行多币种交易"""
    if TRADE_CONFIG['test_mode']:
        print("🧪 测试模式 - 显示交易信号:")
        for signal in signals:
            print(f"  {signal['symbol']}: {signal['signal']} (信心: {signal['confidence']})")
        return
        
    try:
        # 获取当前账户状态
        account_info = get_account_performance()
        current_positions_count = account_info['total_positions']
        max_positions = TRADE_CONFIG['position_management']['max_concurrent_positions']
        
        print(f"📊 账户状态: {current_positions_count}/{max_positions} 个持仓")
        
        # 按信心等级排序信号
        confidence_priority = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        signals.sort(key=lambda x: confidence_priority.get(x['confidence'], 0), reverse=True)
        
        # 执行每个币种的交易信号
        for signal in signals:
            symbol = signal['symbol']
            symbol_full = TRADE_CONFIG['symbols'][symbol]
            
            print(f"\n🎯 处理 {symbol} 信号: {signal['signal']} (信心: {signal['confidence']})")
            print(f"💡 理由: {signal['reason']}")
            
            # 获取当前价格
            ticker = exchange.fetch_ticker(symbol_full)
            current_price = ticker['last']
            print(f"💰 当前价格: {current_price}")
            
            # 获取当前持仓
            current_position = get_current_position_for_symbol(symbol)
            
            # 执行交易逻辑
            if signal['signal'] == 'BUY':
                handle_buy_signal(symbol, symbol_full, current_position, current_price, signal, 
                                current_positions_count, max_positions)
            elif signal['signal'] == 'SELL':
                handle_sell_signal(symbol, symbol_full, current_position, current_price, signal,
                                 current_positions_count, max_positions)
            elif signal['signal'] == 'CLOSE':
                handle_close_signal(symbol, symbol_full, current_position, signal)
            elif signal['signal'] == 'HOLD':
                handle_hold_signal(symbol, current_position, current_price, signal)
            else:
                print(f"❌ 未知信号类型: {signal['signal']}")
                
            # 更新持仓计数
            if signal['signal'] in ['CLOSE'] and current_position:
                current_positions_count -= 1
            elif signal['signal'] in ['BUY', 'SELL'] and not current_position:
                current_positions_count += 1
                
    except Exception as e:
        print(f"交易执行失败: {e}")

def handle_buy_signal(symbol, symbol_full, current_position, current_price, signal_data, 
                     current_positions_count, max_positions):
    """处理买入信号 - 多币种版本"""
    try:
        # 检查持仓限制
        if current_positions_count >= max_positions and not current_position:
            print(f"⚠️ 已达到最大持仓限制({max_positions})，跳过{symbol}开仓")
            return
            
        # 检查现有持仓
        if current_position:
            if current_position['side'] == 'long':
                print(f"✅ {symbol}已有做多持仓")
                # return
            else:
                print(f"🔄 {symbol}存在空头持仓，先平仓")
                close_position(symbol, symbol_full, current_position)
                time.sleep(1)
        
        # 根据信心等级确定仓位乘数
        confidence_multiplier = {
            'HIGH': TRADE_CONFIG['position_management']['high_confidence_multiplier'],
            'MEDIUM': TRADE_CONFIG['position_management']['medium_confidence_multiplier'],
            'LOW': TRADE_CONFIG['position_management']['low_confidence_multiplier']
        }.get(signal_data['confidence'], 1.0)
        
        # 计算基础仓位价值
        base_amount = TRADE_CONFIG['position_management']['base_usdt_amount']
        position_value = base_amount * confidence_multiplier * TRADE_CONFIG['leverage']
        
        print(f"📈 计算仓位 - 基础: {base_amount} USDT, 信心乘数: {confidence_multiplier}, 杠杆: {TRADE_CONFIG['leverage']}x")
        print(f"🎯 实际仓位价值: {position_value} USDT")
        
        # 计算合约数量
        contract_size = get_contract_size(symbol_full)
        contracts = calculate_contracts(symbol_full, position_value, current_price, contract_size)
        
        print(f"🚀 执行{symbol}做多开仓")
        print(f"📊 开仓数量: {contracts} 张, 价格: {current_price}")
        
        # 下开多单
        order_params = {
            'tdMode': 'cross',
            'posSide': 'long',
        }
        
        order = exchange.create_order(
            symbol=symbol_full,
            type='market',
            side='buy',
            amount=contracts,
            params=order_params
        )
        
        print(f"✅ {symbol}做多开仓成功 - 订单ID: {order['id']}")
        
        # 设置止损止盈
        if signal_data.get('stop_loss') and signal_data.get('take_profit'):
            set_stop_loss_take_profit(symbol_full, 'long', contracts, current_price, 
                                    signal_data['stop_loss'], signal_data['take_profit'])
        
    except Exception as e:
        print(f"❌ {symbol}做多开仓失败: {e}")

def handle_sell_signal(symbol, symbol_full, current_position, current_price, signal_data,
                      current_positions_count, max_positions):
    """处理卖出信号 - 多币种版本"""
    try:
        # 检查持仓限制
        if current_positions_count >= max_positions and not current_position:
            print(f"⚠️ 已达到最大持仓限制({max_positions})，跳过{symbol}开仓")
            return
            
        # 检查现有持仓
        if current_position:
            if current_position['side'] == 'short':
                print(f"✅ {symbol}已有做空持仓，无需操作")
                return
            else:
                print(f"🔄 {symbol}存在多头持仓，先平仓")
                close_position(symbol, symbol_full, current_position)
                time.sleep(1)
        
        # 根据信心等级确定仓位乘数
        confidence_multiplier = {
            'HIGH': TRADE_CONFIG['position_management']['high_confidence_multiplier'],
            'MEDIUM': TRADE_CONFIG['position_management']['medium_confidence_multiplier'],
            'LOW': TRADE_CONFIG['position_management']['low_confidence_multiplier']
        }.get(signal_data['confidence'], 1.0)
        
        # 计算基础仓位价值
        base_amount = TRADE_CONFIG['position_management']['base_usdt_amount']
        position_value = base_amount * confidence_multiplier * TRADE_CONFIG['leverage']
        
        print(f"📈 计算仓位 - 基础: {base_amount} USDT, 信心乘数: {confidence_multiplier}, 杠杆: {TRADE_CONFIG['leverage']}x")
        print(f"🎯 实际仓位价值: {position_value} USDT")
        
        # 计算合约数量
        contract_size = get_contract_size(symbol_full)
        contracts = calculate_contracts(symbol_full, position_value, current_price, contract_size)
        
        print(f"🚀 执行{symbol}做空开仓")
        print(f"📊 开仓数量: {contracts} 张, 价格: {current_price}")
        
        # 下开空单
        order_params = {
            'tdMode': 'cross',
            'posSide': 'short',
        }
        
        order = exchange.create_order(
            symbol=symbol_full,
            type='market',
            side='sell',
            amount=contracts,
            params=order_params
        )
        
        print(f"✅ {symbol}做空开仓成功 - 订单ID: {order['id']}")
        
        # 设置止损止盈
        if signal_data.get('stop_loss') and signal_data.get('take_profit'):
            set_stop_loss_take_profit(symbol_full, 'short', contracts, current_price, 
                                    signal_data['stop_loss'], signal_data['take_profit'])
        
    except Exception as e:
        print(f"❌ {symbol}做空开仓失败: {e}")

def handle_close_signal(symbol, symbol_full, current_position, signal_data):
    """处理平仓信号"""
    try:
        if not current_position:
            print(f"💤 {symbol}无持仓，无需平仓")
            return
            
        print(f"🔄 执行{symbol}平仓")
        close_position(symbol, symbol_full, current_position)
        
    except Exception as e:
        print(f"❌ {symbol}平仓失败: {e}")

def handle_hold_signal(symbol, current_position, current_price, signal_data):
    """处理持有信号 - 当信号为HOLD时不进行任何处理"""
    # 当信号为HOLD时，直接返回，不执行任何操作
    print(f"💤 {symbol} 信号为HOLD，不进行任何处理")
    return

def close_position(symbol, symbol_full, position):
    """平仓指定币种的持仓"""
    try:
        if position['side'] == 'long':
            close_side = 'sell'
            pos_side = 'long'
        else:  # short
            close_side = 'buy' 
            pos_side = 'short'
            
        order_params = {
            'tdMode': 'cross',
            'posSide': pos_side,
            'reduceOnly': True
        }
            
        order = exchange.create_order(
            symbol=symbol_full,
            type='market',
            side=close_side,
            amount=position['size'],
            params=order_params
        )
        
        print(f"✅ {symbol}平仓成功 - 订单ID: {order['id']}")
        return order
        
    except Exception as e:
        print(f"❌ {symbol}平仓失败: {e}")
        raise e

def get_contract_size(symbol_full):
    """获取合约面值 - 修复版本"""
    try:
        # 首先尝试直接加载市场
        markets = exchange.load_markets()
        
        # 尝试多种可能的符号格式
        possible_symbols = [
            symbol_full,  # 原始符号: BTC-USDT-SWAP
            symbol_full.replace('-SWAP', ':USDT'),  # BTC-USDT:USDT
            symbol_full.replace('-USDT-SWAP', '/USDT:USDT'),  # BTC/USDT:USDT
        ]
        
        for symbol in possible_symbols:
            if symbol in markets:
                symbol_info = markets[symbol]
                contract_size = float(symbol_info['contractSize'])
                print(f"📏 {symbol_full}合约面值: {contract_size} (使用符号: {symbol})")
                return contract_size
        

        
    except Exception as e:
        print(f"❌ 获取{symbol_full}合约面值失败: {e}")

def calculate_contracts(symbol_full, position_value, current_price, contract_size):
    """计算合约张数 - 简化版本（直接向下取整）"""
    try:
        # 计算基础合约数量
        contracts = position_value / (current_price * contract_size)
        
        # 🎯 直接向下取整
        contracts = int(contracts)  # 3.679 → 3
        
        # 确保至少1张
        contracts = max(contracts, 1)
        
        print(f"📊 {symbol_full}合约计算:")
        print(f"   - 仓位价值: {position_value} USDT")
        print(f"   - 当前价格: {current_price}")
        print(f"   - 合约面值: {contract_size}")
        print(f"   - 最终数量: {contracts}张")
        
        return contracts
        
    except Exception as e:
        print(f"❌ 计算{symbol_full}合约数量失败: {e}")
        return 1  # 返回安全值

def check_account_mode():
    """检查账户模式"""
    try:
        print("🔍 检查账户模式...")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
        print(f"💰 账户余额: {usdt_balance} USDT")
        
        return True
    except Exception as e:
        print(f"❌ 检查账户模式失败: {e}")
        return False

def set_stop_loss_take_profit(symbol_full, position_side, contracts, entry_price, stop_loss, take_profit):
    """设置止损止盈"""
    try:
        # 这里可以根据交易所API实现止损止盈设置
        print(f"🛡️ {symbol_full}止损止盈设置 - 止损: {stop_loss}, 止盈: {take_profit}")
        # 实际实现需要根据交易所API进行调整
        
    except Exception as e:
        print(f"⚠️ {symbol_full}止损止盈设置失败: {e}")

def check_position_risk(symbol, position, current_price):
    """检查持仓风险"""
    try:
        if position['side'] == 'long':
            price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
        else:  # short
            price_change = ((position['entry_price'] - current_price) / position['entry_price']) * 100
            
        leverage = position.get('leverage', TRADE_CONFIG['leverage'])
        actual_pnl_percent = price_change * leverage
        
        print(f"📊 {symbol}风险分析:")
        print(f"   - 价格变动: {price_change:.2f}%")
        print(f"   - 杠杆效应: {actual_pnl_percent:.2f}%")
        
        # 风险等级评估
        if abs(actual_pnl_percent) > 50:
            risk_level = "🔴 极高风险"
        elif abs(actual_pnl_percent) > 30:
            risk_level = "🟠 高风险"
        elif abs(actual_pnl_percent) > 15:
            risk_level = "🟡 中等风险"
        else:
            risk_level = "🟢 低风险"
            
        print(f"   - 风险等级: {risk_level}")
        
    except Exception as e:
        print(f"⚠️ {symbol}风险检查失败: {e}")

# 在trading_bot函数中添加余额记录（每次执行周期记录）
def trading_bot():
    """主交易机器人函数 - 多币种版本"""
    global execution_count, balance_history  # 引用全局变量
    
    print(f"\n🎯 第{execution_count + 1}次执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录当前账户余额
    total_balance, _ = get_account_balance()
    balance_history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'balance': total_balance
    })
    print(f"📝 记录账户余额: {total_balance:.2f} USDT (历史记录数: {len(balance_history)})")
    
    # 使用DeepSeek Reasoning模型分析
    signals = analyze_with_deepseek_reasoning()
    
    # 执行交易
    execute_trades(signals)
    
    # 保存信号历史
    for signal in signals:
        signal['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        signal_history.append(signal)
    
    print(f"✅ 多币种推理分析完成 - 处理了{len(signals)}个币种")

def main():
    """主函数"""
    print("🚀 多币种OKX自动交易机器人启动成功！")
    print("🧠 使用DeepSeek Reasoning模型进行深度推理分析")
    print(f"📊 交易币种: {', '.join(TRADE_CONFIG['symbols'].keys())}")
    
    if not setup_exchange():
        print("❌ 交易所初始化失败，程序退出")
        return
        
    # 立即执行一次
    trading_bot()

    # 每2分钟执行一次
    schedule.every(2).minutes.do(trading_bot)

    print("⏰ 程序开始运行，每2分钟执行一次多币种推理分析...")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()