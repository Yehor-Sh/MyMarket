import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# Конфигурация - ослабляем фильтры для тестирования
MIN_QV_USDT = 10000000  # Уменьшили объем для тестирования
MIN_PRICE = 0.3        # Расширили диапазон цен
MAX_PRICE = 5000.0
TIMEFRAMES = ['5m', '15m', '30m']  # Уменьшили количество таймфреймов

def fetch_klines(symbol, interval, limit=100):
    """Получение свечных данных с Binance"""
    try:
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Конвертация числовых значений
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
        
        return df
    except Exception as e:
        print(f"Ошибка получения данных для {symbol} {interval}: {e}")
        return None

def get_all_symbols():
    """Получение всех торговых пар с USDT"""
    try:
        response = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=10)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] 
                  if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT']
        return symbols
    except Exception as e:
        print(f"Ошибка получения списка символов: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']  # Fallback

def detect_reversal_patterns(df):
    """Обнаружение разворотных паттернов"""
    if df is None or len(df) < 10:
        return None
    
    patterns = []
    
    # Берем последние 3 свечи для анализа
    current = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Определяем тренд предыдущих свечей
    prev_trend = "BULL" if prev1['close'] > prev1['open'] else "BEAR"
    
    # 1. Поглощающая свеча (Engulfing)
    is_bull_engulfing = (current['close'] > current['open'] and 
                        prev1['close'] < prev1['open'] and
                        current['open'] < prev1['close'] and
                        current['close'] > prev1['open'])
    
    is_bear_engulfing = (current['close'] < current['open'] and 
                        prev1['close'] > prev1['open'] and
                        current['open'] > prev1['close'] and
                        current['close'] < prev1['open'])
    
    # 2. Молот (Hammer) и повешенный (Hanging Man)
    body = abs(current['close'] - current['open'])
    total_range = current['high'] - current['low']
    lower_shadow = min(current['open'], current['close']) - current['low']
    upper_shadow = current['high'] - max(current['open'], current['close'])
    
    is_hammer = (lower_shadow >= 2 * body and 
                upper_shadow <= 0.1 * total_range and
                current['close'] > current['open'])
    
    is_hanging_man = (lower_shadow >= 2 * body and 
                     upper_shadow <= 0.1 * total_range and
                     current['close'] < current['open'])
    
    # 3. Паттерн "Утренняя звезда" и "Вечерняя звезда"
    is_morning_star = (prev2['close'] < prev2['open'] and
                      abs(prev1['close'] - prev1['open']) / prev1['open'] < 0.01 and  # Доджи
                      current['close'] > current['open'] and
                      current['close'] > prev2['open'] * 0.5)
    
    is_evening_star = (prev2['close'] > prev2['open'] and
                      abs(prev1['close'] - prev1['open']) / prev1['open'] < 0.01 and  # Доджи
                      current['close'] < current['open'] and
                      current['close'] < prev2['open'] * 1.5)
    
    # Формируем сигналы
    if is_bull_engulfing and prev_trend == "BEAR":
        patterns.append(('LONG', 'ENGULFING'))
    
    if is_bear_engulfing and prev_trend == "BULL":
        patterns.append(('SHORT', 'ENGULFING'))
    
    if is_hammer and prev_trend == "BEAR":
        patterns.append(('LONG', 'HAMMER'))
    
    if is_hanging_man and prev_trend == "BULL":
        patterns.append(('SHORT', 'HANGING_MAN'))
    
    if is_morning_star:
        patterns.append(('LONG', 'MORNING_STAR'))
    
    if is_evening_star:
        patterns.append(('SHORT', 'EVENING_STAR'))
    
    return patterns

def get_signals():
    """Генератор сигналов - отдает сигналы по одному по мере обнаружения"""
    # Получаем список символов
    symbols = get_all_symbols()
    print(f"Анализируем {len(symbols)} символов")
    
    # Ограничиваем количество для тестирования
    test_symbols = symbols[:20]  # Первые 20 символов для теста
    
    for symbol in test_symbols:
        try:
            # Получаем информацию об объеме
            ticker_url = f'https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}'
            ticker_data = requests.get(ticker_url, timeout=5).json()
            
            if 'quoteVolume' not in ticker_data:
                continue
                
            quote_volume = float(ticker_data['quoteVolume'])
            last_price = float(ticker_data['lastPrice'])
            
            # Проверяем фильтры
            if (quote_volume < MIN_QV_USDT or 
                last_price < MIN_PRICE or 
                last_price > MAX_PRICE):
                continue
            
            # Анализируем на разных таймфреймах
            for timeframe in TIMEFRAMES:
                df = fetch_klines(symbol, timeframe, 20)
                if df is None or len(df) < 10:
                    continue
                
                # Ищем паттерны
                patterns = detect_reversal_patterns(df)
                
                if patterns:
                    for direction, pattern in patterns:
                        current = df.iloc[-1]
                        
                        signal = {
                            'symbol': symbol,
                            'signal': direction,
                            'entry': current['close'],
                            'sl': current['low'] * 0.98 if direction == 'LONG' else current['high'] * 1.02,
                            'timeframe': timeframe,
                            'pattern': pattern,
                            'volume': quote_volume,
                            'price': last_price,
                            'time': datetime.now().strftime('%H:%M:%S')
                        }
                        
                        print(f"Найден сигнал: {symbol} {timeframe} {direction} {pattern}")
                        yield signal  # Отправляем сигнал сразу при обнаружении
            
        except Exception as e:
            print(f"Ошибка анализа {symbol}: {e}")
            continue

# Для обратной совместимости оставляем старую функцию
def get_signals_list():
    """Функция для обратной совместимости с app.py"""
    return list(get_signals())

# Для тестирования модуля
if __name__ == "__main__":
    while True:
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Поиск сигналов...")
        signals = list(get_signals())  # Преобразуем в список для совместимости
        print(f"Найдено сигналов: {len(signals)}")
        
        for signal in signals:
            print(f"{signal['symbol']} {signal['signal']} {signal['pattern']} {signal['timeframe']}")
        
        time.sleep(60)  # Проверяем каждую минуту