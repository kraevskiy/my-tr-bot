# SMC-бот для EURUSD / GBPUSD — v3.2 (Strategy v6)

Бот формалізує методологію курсу (ICT/SMC) у систему, яку можна перевірити на історії. Два сетапи та правило кореляції:
**v4** — свіп ліквідності → зсув структури на M5 і M3 → лімітний вхід у FVG; **AF** — фейковий пробій діапазону Азії
на відкритті Лондона → вхід проти пробою; **правило кореляції** — якщо інша пара вже в позиції в той самий бік, новий сигнал береться з половинним ризиком замість блокування.
Стоп/ціль/ризик за правилами PRO, усе проганяється на M1-даних брокера з урахуванням спреду.

- **Правила стратегії** → [STRATEGY.md](STRATEGY.md) (Strategy v6 = v5 + друга пара ×0.5, працює за замовчуванням; v5 = v4 + AF)
- **Як влаштований код** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Історія версій і досліджень** → [CHANGELOG.md](CHANGELOG.md)

| Бектест 2024-01 → 2026-10, M1 MT5, EURUSD + GBPUSD | v4 сам | v5 (v4 + AF) | **v6 (v5 + друга пара ×0.5)** |
|---|---|---|---|
| Угод | 104 (~3/міс), вінрейт 58% | 273 (~8/міс), вінрейт 47% | **286** (~8/міс), вінрейт 48% |
| Результат у базовому ризику | +99R, просадка 2.1R | +154R, просадка 3.5R (Monte Carlo 95%: 8R) | **+158R**, просадка 3.5R (Monte Carlo 95%: 8.2R) |
| Прибуткових місяців | 26 з 34 | 31 з 34 | **31 з 34** |
| Реалістично при ризику 0.5% | ~12–15%/рік | ~12–18%/рік | **~12–18%/рік** |

v6 статистично нерозрізнювана з v5 (+4.1 Rw, 14 угод із половинним ризиком): це вибір ризик-політики, а не нового сигналу.

> **Лише ECN/raw рахунок:** бектест на сирому спреді MT5 без комісії; спред +1 пп з'їдає більшу частину переваги v4.

> Це результат на історії, а не гарантія. Перед реальними грошима — форвард-тест на демо.

---

## 1. Швидкий старт

### Mac / Linux — Docker

```bash
cp .env.example .env              # DATA_SOURCE=csv, STRATEGY=v6
# покладіть експорт з MT5 у ./data:  EURUSD_M1.csv, GBPUSD_M1.csv
docker compose build
docker compose run --rm backtest
```

Як вивантажити історію з MT5: відкрийте **Сервіс → Символи** (Ctrl+U) → виберіть символ → вкладка **Бари** →
таймфрейм **M1**, період → **Запит** → **Експорт барів**. Кілька файлів на символ (частинами) теж підходять:
`CSV_EURUSD=/app/data/EURUSD_M1*.csv`.

### Windows — напряму з терміналу MT5

1. Python 3.11–3.13 з python.org («Add to PATH»).
2. У MT5: Сервіс → Налаштування → Графіки → «Макс. барів у вікні» = Unlimited; EURUSD і GBPUSD в «Огляді ринку»; термінал запущений.
3. Подвійний клік по `run_windows.bat` (перший запуск створить `.venv` і `.env`; у `.env` поставте `DATA_SOURCE=mt5`).
   Інші команди: `run_windows.bat experiments`, `run_windows.bat analyze results\<run> V4` тощо.

### Без Docker (будь-яка ОС)

```bash
pip install -e .                  # або: pip install -r requirements.txt
smcbot backtest                   # те саме, що python -m smcbot backtest
```

## 2. Команди

Одна точка входу: `python -m smcbot <команда>` (у Docker — `docker compose run --rm <команда>`).

| Команда | Що робить | Результат |
|---|---|---|
| `backtest` | один прогін стратегії з `.env` | `results/summary.txt`, `signals.csv`, `trades_c.csv` |
| `experiments [--only A,B] [--modes b] [--file f.json]` | багато варіантів з `experiments.json` за раз | `results/<run_id>/experiments.csv`, `report.md`, угоди та HTML-звіт по кожному варіанту з `"log": true` |
| `analyze <run_dir> [variant] [--best]` | глибокий HTML-звіт: t-stat, bootstrap, Monte Carlo, розрізи, MFE/MAE, walk-forward, новини, класи A/B | `<run>/<variant>/analysis_b.html` |
| `news` | оновити календар ForexFactory (дзеркало, UTC) | `data/news/ff_calendar.csv` |
| `sessions` | дослідження Азія → Франкфурт → Лондон → NY | `results/sessions_<SYMBOL>.csv` |
| `presets` | список готових стратегій | — |
| `config` | показати фактичну конфігурацію (пресет + `.env`) | — |
| `neighbours [--n 60 --pct 0.2 --seed 7 --out f.json]` \| `--summarize <run_dir>` | випадкові сусіди пресету ±20 % для `experiments --file`; підсумок і вердикти робастності | `f.json`; вердикти в консоль |

Тести: `docker compose run --rm tests` або `pytest`.

## 3. Налаштування

Усе задається в `.env` (повний список з поясненнями — у `.env.example`):

| Змінна | За замовч. | Що задає |
|---|---|---|
| `STRATEGY` | `v6` | готова стратегія: `course`, `v1`…`v4`, `v5_core` (v4 + покращення, без AF), `v5` (v4 + AF), **`v6`** (v5 + друга пара ×0.5), `af_only`, `v3_asia` |
| `CFG_AF_RISK_MULT` | `0.5` | ризик угоди AF відносно базового |
| `CFG_KZ_RISK_MULT`, `CFG_AF_SYMBOLS`, `CFG_BLOCK_SAME_DIRECTION` | `{}`, `[]`, `true` | множник ризику за кілзоною, AF лише на вказаних парах, заборона двох пар в один бік одночасно; напр. `CFG_KZ_RISK_MULT={"ASIA":0.5}`, `CFG_AF_SYMBOLS=["EURUSD"]`, `CFG_BLOCK_SAME_DIRECTION=false` |
| `CFG_CORR_SECOND_RISK_MULT`, `CFG_FILL_THROUGH_PIPS` | `0.5` (у v6; `0` у v5), `0` | друга пара в той самий бік замість блокування — з ризиком ×X (напр. `0.5`; 0 = блокувати); стрес-тест виконання: ліміт заповнюється лише після проходу ціни крізь рівень на X піпсів (напр. `0.2`) |
| `CFG_<ПОЛЕ>` | — | будь-яке поле `smcbot/config.py` поверх пресету, напр. `CFG_MIN_SL_PIPS=6`, `CFG_GRADE_MODE=a_only` |
| `DATA_SOURCE` | `csv` | `csv` (експорт MT5), `mt5` (термінал, лише Windows), `dukascopy` |
| `CSV_EURUSD`, `CSV_GBPUSD` | `/app/data/<SYM>_M1*.csv` | шляхи / маски до файлів |
| `MT5_SERVER_TZ` | `ny_close` | часовий пояс сервера брокера (більшість — GMT+2/+3 з переходом, як NY close) |
| `SYMBOLS`, `DATE_FROM`, `DATE_TO` | `EURUSD,GBPUSD`, `2024-01-01`, сьогодні | що і за який період |
| `RISK_PCT`, `DAILY_MAX_LOSS_PCT`, `DAILY_MAX_SL` | `0.3`, `1.5`, `5` | ризик PRO |
| `BE_MODES` | `c` | режим стопу: `c` (v5) — після часткової 2R у беззбиток на новому M5-свінгу; також `a`/`b`/`d` |
| `OOS_FROM` | `2026-01-01` | межа in-sample / out-of-sample у звітах |

Свої варіанти для перевірки — в `experiments.json`: `"preset"` + список `variants` з будь-якими `params`.

## 4. Структура репозиторію

```
smcbot/
  config.py  settings.py        параметри стратегії, пресети v1–v6, читання .env
  cli.py  __main__.py           python -m smcbot <команда>
  data/                         bars (TF, часові пояси), providers (csv/mt5/dukascopy), synthetic
  market/                       sessions, structure, zones (FVG/OB), liquidity, smt, news (календар, свята)
  strategy/setups.py            сетап v4: свіп → Shift → вхід, фільтри, клас A/B
  strategy/asia_fake.py         сетап AF: фейк Азії на відкритті Лондона
  engine/backtest.py            симуляція на M1, стоп-менеджмент, ліміти PRO, метрики
  research/                     analytics (HTML-звіти), sessions (зв'язок сесій)
  apps/                         реалізація команд (neighbours.py — сусіди пресету та вердикти)
news/ff_calendar.csv            вшитий календар (USD/EUR/GBP, 2023–2026)
tests/                          81 тест (структура, зони, без заглядання в майбутнє, угоди, новини, класи, AF…)
experiments.json                поточний раунд перевірок (v5: v4 vs v5_core vs v5, ризик AF)
experiments_r11.json            раунд 11 (V5_CORE, H1…H7)
experiments_r12.json            раунд 12 (стрес виконання, блок K, corr_second_risk_mult)
RESEARCH_LOG.md                 журнал досліджень з раунду 11
STRATEGY.md  ARCHITECTURE.md  CHANGELOG.md
```

Старі скрипти `run_backtest.py`, `run_experiments.py`, `analyze.py`, `fetch_news.py`, `study_sessions.py` залишені для сумісності
і просто викликають відповідні команди.

## 5. Індикатор для TradingView

`tradingview/smc_v5.pine` — обидва сетапи на графіку: на M5 — сигнали, лінії, результат кожної угоди і статистика окремо v4 / AF;
на H1 і вище — мітки входів з результатом; сповіщення TradingView. Як встановити — `tradingview/README.md`.

## 6. Що далі

- Форвард-тест v6 на демо (2–3 місяці): AF з ризиком ×0.5, поки не підтвердиться ~+0.4R/угоду; друга пара в той самий бік — з ризиком ×0.5; сповіщення з індикатора або Telegram-бот.
- Агресивніша альтернатива — H7 (`CFG_BLOCK_SAME_DIRECTION=false`: друга пара з повним ризиком): пройшла правила раунду 11 (+8.5 Rw), але це зміна ризик-політики, не сигналу; сумарна експозиція при двох лонгах ≈ 1.8σ проти ≈ 1.38σ у v6. Блок → ×0.5 → повний розмір: кожен крок ≈ +2 Rw і ≈ +0.3R просадки на медіані сусідів; v6 — збалансована точка.
- Стрес реалістичності виконання коштує 3–8 % результату (−4.7 Rw при 0.2 піпса, −12.2 Rw при 0.5), тож очікування для живої торгівлі слід відповідно зменшити.
- Ті самі правила на інших інструментах (USDJPY, GBPJPY, AUDUSD, USDCAD, XAUUSD) — найсильніша перевірка, що перевага справжня.

Реалістичні очікування від v5 — медіана сусідів (≈ +117 Rw за 34 міс., DD 5–6R), а не пік бектесту; для v6 медіана сусідів ≈ +119 Rw / 34 міс., DD ≈ 5.7R (p95 8.2R).
