# Agent Specification

## Overview
Проект разделён на агентов, каждый из которых отвечает за свой слой:
- получение и кеширование маркет-данных,
- генерация сигналов стратегий,
- агрегирование сигналов в кластеры,
- управление стратегиями через воркеры,
- координация сделок и трейлинг-стопов,
- UI для наблюдения за системой.

## Agents

### 1. Binance Client (`binance_client.py`)
- **Role:** Источник данных (REST + WebSocket).
- **Input:** запросы на цены и свечи.
- **Output:** тикеры, свечи, ликвидные пары.
- **Dependencies:** используется модулями стратегий и оркестратором.

### 2. Strategy Modules (`modules/`)
- **Role:** Аналитика свечей → торговые сигналы.
- **Base class:** `ModuleBase` (`module_base.py`).
- **Signal format:**
  ```python
  Signal(symbol: str, side: Literal["LONG","SHORT"], strategy: str, confidence: float, metadata: dict)
  ```
- **Examples (см. `multi_timeframe_config.py`):**
  - DIV — RSI Divergence
  - PIN — Pin Bar + Level + EMA
  - ENG — Engulfing + RSI
  - BRK — ATR + EMA Breakout
  - INS — Inside Bar Breakout

### 2a. Cluster Engine (`modules/cluster_engine.py`)
- **Role:** Группирует сигналы стратегий по `(symbol, side)` и выпускает
  подтверждённые кластеры при достижении порога стратегий.
- **Output:** агрегированные сигналы со средним `confidence` и
  метаданными `cluster_size`, `strategies`, `source_signals`.

### 3. Module Worker (`module_worker.py`)
- **Role:** Запускает стратегию в отдельном потоке.
- **Input:** модуль стратегии, список символов.
- **Output:** сигналы в очередь.
- **Extras:** отдаёт health-статус (idle, running, ok, error).

### 4. Orchestrator (`orchestrator.py`)
- **Role:** Главный управляющий.
- **Input:** сигналы из очереди, цены от Binance Client.
- **Output:** сделки + обновления для UI через WebSocket.
- **Functions:**
  - trailing stop (по умолчанию 0.3%),
  - активные и закрытые сделки,
  - маршрутизация сигналов в трейды.

### 5. Dashboard (`client.html`)
- **Role:** Интерфейс наблюдения.
- **Input:** WebSocket события от Orchestrator.
- **Output:** отображение сделок, стратегий, здоровья модулей.

## Interaction Flow
1. **Binance Client** обновляет тикеры и свечи.
2. **Strategy Modules** берут данные и генерируют `Signal`.
3. **Module Workers** гоняют стратегии циклически и кидают сигналы в очередь.
4. **Orchestrator** обрабатывает сигналы → открывает/ведёт сделки → применяет трейлинг.
5. **Dashboard** через WebSocket видит текущее состояние.
