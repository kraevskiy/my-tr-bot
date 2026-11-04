# Архітектура

## Шари

Кожен шар залежить лише від шарів вище. Ринкові об'єкти не знають про стратегію; стратегія не знає, як виконуються угоди.

```
config.py / settings.py   Config (усі параметри), Instrument, PRESETS (course, v1–v6), .env → Config
        │
data/                     M1-бари → New York time → таймфрейми M3…W1
        │
market/                   sessions · structure · zones · liquidity · smt · news
        │                 (чисті функції від історії; кожен об'єкт має known_at)
strategy/setups.py        SymbolModels → SideModel(long) + SideModel(short, дзеркало)  [+ AF, якщо cfg.asia_fake]
strategy/asia_fake.py     asia_fake_signals(m1) → ринкові сигнали AF (фейк Азії)
        │                 етап A: кандидати (без параметрів) · етап B: варіант Config → сигнали
engine/backtest.py        simulate_trade (M1, спред, часткова, стоп-режими) · run_backtest (ліміти PRO) · metrics
        │
research/                 analytics (HTML-звіт) · sessions (дослідження сесій)
        │
apps/ + cli.py            backtest · experiments · analyze · news · sessions · prop · neighbours · presets · config
```

## Потік даних

```
CSV / MT5 / Dukascopy ──► data.providers.load_symbol ──► M1 DataFrame (NY time, bid + spread)
                                                     │
                     strategy.setups.SymbolModels(m1, cfg, instrument)
                     ├─ data.bars.build_timeframes            M3, M5, M15, H1, H4, D1, W1
                     ├─ market.structure.market_structure     тренд (BOS/Conf) на M15…W1
                     ├─ market.zones.find_fvgs / find_obs     POI на M15/H1/H4, FVG на M5
                     ├─ market.liquidity.build_pools          пули Major/Medium/Minor + час зняття
                     └─ find_shifts(M5), find_shifts(M3)      зсуви структури
                                                     │
              .signals(cfg) ──► сигнали (entry, sl, tp, setup, grade, risk_mult, …)
                                                     │
              market.news: holiday_filter / news_filter / annotate
                                                     │
              engine.backtest.run_backtest ──► trades (R_result, R_w, MFE/MAE, exit_kind, …) ──► metrics / analytics
```

## Ключові рішення

**Без заглядання в майбутнє.** Кожен об'єкт ринку має час, коли він став відомим: свінг — після закриття
`n`-го бару праворуч, FVG — після закриття третьої свічки, пул — після свого періоду (Asia H/L — о 02:00 NY),
бари старшого ТФ — лише закриті. Сигнал використовує тільки те, що відоме на момент закриття бару сигналу.
Окремий тест перевіряє, що обрізання майбутніх даних не змінює минулі сигнали.

**Одна логіка для лонгів і шортів.** Шорти рахуються тим самим кодом на дзеркальній копії ринку
(`open/high/low/close → −open/−low/−high/−close`); результат перетворюється назад (`unmirror`). Це прибирає цілий клас помилок «забули поміняти знак».

**Два етапи сигналів.** Етап A (`base_candidates`) один раз знаходить усі Shift на M5 з ознаками, які не залежать
від параметрів: знятий пул, синхрон з M3, FVG/OB, нога імпульсу. Етап B (`signals(cfg)`) застосовує до них конкретний
варіант: вікна, фільтри, вхід, стоп, ціль, клас. Тому 20 варіантів у `experiments` займають хвилини, а не години.
Кожна причина відсіву записується (`funnel`, `candidates.csv.gz`).

**Консервативне виконання.** Симуляція на M1: лімітка виконується лише якщо ask/bid реально торкнувся ціни;
якщо в одному барі зачеплено і стоп, і ціль — рахуємо стоп; спред з даних брокера; опція `slippage_pips` для стрес-тесту.
Ордер скасовується, якщо ціна дійшла до тейку раніше за вхід.

**Конфігурація як дані.** Усе, що можна змінити, — поле `Config`. Пресети (`PRESETS`) — це просто словники змін,
`.env` (`STRATEGY`, `CFG_*`) і `experiments.json` накладаються поверх. Варіант стратегії = JSON, без зміни коду.

## Модулі

| Модуль | Відповідальність |
|---|---|
| `config.py` | `Config` (вікна, фільтри, вхід, вихід, ризик, класи, новини), `Instrument`, `PRESETS`, `killzones()` |
| `settings.py` | `.env` → `Settings` (джерело даних, шляхи) + `Config` (пресет, `CFG_*`) |
| `data/bars.py` | читання MT5 CSV (кодування, час сервера → NY), торговий день 17:00 NY, ресемплінг, `drop_weekend` |
| `data/providers.py`, `data/mt5.py` | джерела `csv` / `dukascopy` (кеш по місяцях) / `mt5` (Windows) |
| `data/synthetic.py` | синтетичні M1 для тестів і швидких перевірок |
| `market/sessions.py` | killzones, Азія, CBDR, денні/тижневі/місячні рівні, NY-midnight open |
| `market/structure.py` | фрактальні свінги, BOS/Conf за закриттями, рівні свінгів з `known_at` |
| `market/zones.py` | FVG і OB з часом появи та інвалідації |
| `market/liquidity.py` | пули (PDH/PDL, PWH/PWL, PMH/PML, Asia, London, свінги H1/M15) і перше зняття |
| `market/smt.py` | SMT-дивергенція EURUSD ↔ GBPUSD |
| `market/news.py` | календар ForexFactory, свята, контекст новин для угод, фільтри новин і свят |
| `strategy/setups.py` | `find_shifts`, `SideModel` (кандидати, фільтри, вхід, стоп, ціль, клас A/B), `SymbolModels` (додає сигнали AF) |
| `strategy/asia_fake.py` | `build_m5`, `asia_fake_signals` — сетап AF; ринковий ордер, `risk_mult = af_risk_mult`, без часткової |
| `engine/backtest.py` | `simulate_trade`, `run_backtest` (денні ліміти, 1 позиція на пару, кореляція; заблоковані сигнали лишаються в угодах зі статусами `blocked_*` і `shadow_R`; правило кореляції пар: блок / повний ризик / зменшений — `corr_second_risk_mult`; стрес виконання — `fill_through_pips`; беззбиток AF — `af_be_at_r`, вихід `SL_BE`, за замовчуванням вимкнено), `metrics`, `summarize` |
| `research/analytics.py` | HTML-звіт: статистика, bootstrap, Monte Carlo, розрізи, walk-forward, графіки |
| `research/sessions.py` | таблиця сесій по днях і зв'язки між ними |
| `apps/neighbours.py` | `neighbours`: випадкові сусіди пресету ±pct → experiments JSON; `--summarize` — вердикти робастності |
| `apps/prop.py` | `prop`: симулятор двофазного проп-челенджу (rolling-старти і bootstrap по місяцях, лише закриті угоди, `--mae`) |
| `apps/*.py`, `cli.py` | команди |

## Як додати нову ідею

1. Поле в `Config` (за замовчуванням — вимкнено, щоб старі результати не змінились).
2. Ознака, яка не залежить від параметрів → етап A (`base_candidates`); правило/фільтр → етап B (`signals`), з причиною відсіву.
3. Нова колонка в рядку сигналу — вона автоматично з'явиться в `signals.csv`, `trades_*.csv` і в розрізах звіту.
4. Варіант у `experiments.json` (для раунду — окремий `experiments_r<N>.json`) → `python -m smcbot experiments --only V4,МІЙ_ВАРІАНТ`.
5. Тест у `tests/`.
6. Перевірка сусідами: `python -m smcbot neighbours` → `experiments --file` → `neighbours --summarize <run_dir>`.
