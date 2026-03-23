"""
=============================================================================
  HYBRID S&P 500 INVESTMENT MODEL
=============================================================================
  Combina lo mejor de dos modelos de YouTubers:

  MODEL 1 (AI-portfolio) — Modelo ML con LightGBM:
    - 50+ indicadores técnicos (RSI, MACD, Bollinger Bands, ATR, ROC, etc.)
    - LightGBM con CPCV cross-validation (evita overfitting)
    - Walk-forward training con purge & embargo gaps
    - Feature selection: Boruta, PCA, correlación

  MODEL 2 (AI.2-portfolio) — Estrategia MACD pura:
    - MACD histograma cruza de negativo a positivo
    - MACD < 0 y Signal < 0 (momentum temprano)
    - Precio > EMA200 (confirmación de tendencia alcista)
    - Stop Loss 2%, Take Profit 3%

  HYBRID — Ensemble de ambas señales:
    - 60% peso al modelo ML + 40% al MACD
    - Entra cuando el score combinado supera 0.55
    - Hereda el risk management del Modelo 2

  BENCHMARK: SPY (ETF del S&P 500) — Buy & Hold

  MÉTRICAS: CAGR, Sharpe Ratio, Max Drawdown
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'figures'
SPY_PATH   = '../AI-portfolio/Data/SPY.csv'

# Risk management (tomado directamente del AI.2-portfolio)
STOP_LOSS_PCT   = 0.02   # 2% stop loss
TAKE_PROFIT_PCT = 0.03   # 3% take profit (ratio 1:1.5)

# Pesos del ensemble
W_MODEL1 = 0.60   # peso del modelo ML
W_MODEL2 = 0.40   # peso del modelo MACD
ENTRY_THRESHOLD = 0.55   # score mínimo para entrar en posición

# ─────────────────────────────────────────────────────────────
#  1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def load_spy_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV de SPY del AI-portfolio.
    Formato: Date, Close/Last, Volume, Open, High, Low
    Las fechas vienen en formato MM/DD/YYYY y los precios con signo $.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    for col in ['Close/Last', 'Open', 'High', 'Low']:
        df[col] = (
            df[col].astype(str)
            .str.replace('$', '', regex=False)
            .str.strip()
            .astype(float)
        )

    df = df.rename(columns={
        'Close/Last': 'close', 'Open': 'open',
        'High': 'high',        'Low':  'low',
        'Volume': 'volume'
    })

    df = (df[['date', 'open', 'high', 'low', 'close', 'volume']]
          .sort_values('date')
          .reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────
#  2. INDICADORES TÉCNICOS (inspirados en AI-portfolio)
# ─────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI — Relative Strength Index (usado en AI-portfolio)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD — Moving Average Convergence Divergence (ambos modelos lo usan)."""
    ema_f  = series.ewm(span=fast,   adjust=False).mean()
    ema_s  = series.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal, adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist


def _bollinger(series: pd.Series, period=20, n_std=2):
    """Bollinger Bands — ancho y posición dentro de las bandas (AI-portfolio)."""
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    bb_width = (upper - lower) / sma.replace(0, np.nan)
    bb_pos   = (series - lower) / (upper - lower).replace(0, np.nan)
    return bb_width, bb_pos


def _atr(df: pd.DataFrame, period=14) -> pd.Series:
    """ATR — Average True Range, mide la volatilidad real (AI-portfolio)."""
    prev_c = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_c).abs(),
        (df['low']  - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos.
    Implementa el subconjunto más relevante del FeatureEngine del AI-portfolio:
      - Momentum:   RSI, ROC, MOM, Stochastic
      - Tendencia:  MACD, EMA9/20/50/200, SMA20/50
      - Volatilidad: ATR, NATR, Bollinger Bands
      - Volumen:    z-score de volumen
      - Retornos:   log-returns y forward return (target del ML)
    """
    c = df['close']

    # — MACD (idéntico al AI.2-portfolio) —
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = _macd(c)

    # — EMAs de tendencia —
    for span in [9, 12, 26, 50, 200]:
        df[f'EMA{span}'] = c.ewm(span=span, adjust=False).mean()
    df['trend_up'] = (c > df['EMA200']).astype(int)

    # — SMAs —
    for win in [20, 50]:
        df[f'SMA{win}'] = c.rolling(win).mean()

    # — RSI —
    df['RSI_14'] = _rsi(c, 14)
    df['RSI_7']  = _rsi(c,  7)

    # — Bollinger Bands —
    df['BB_width'], df['BB_pos'] = _bollinger(c)

    # — ATR / NATR —
    df['ATR_14'] = _atr(df, 14)
    df['NATR']   = df['ATR_14'] / c.replace(0, np.nan) * 100

    # — ROC — Rate of Change —
    df['ROC_5']  = c.pct_change(5)  * 100
    df['ROC_20'] = c.pct_change(20) * 100

    # — Precio relativo a medias —
    df['price_vs_sma20'] = (c - df['SMA20']) / df['SMA20'].replace(0, np.nan) * 100
    df['price_vs_sma50'] = (c - df['SMA50']) / df['SMA50'].replace(0, np.nan) * 100

    # — MOM — Momentum absoluto —
    df['MOM_10'] = c - c.shift(10)
    df['MOM_20'] = c - c.shift(20)

    # — Stochastic %K / %D —
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    denom  = (high14 - low14).replace(0, np.nan)
    df['STOCH_K'] = 100 * (c - low14) / denom
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()

    # — Volumen z-score (AI-portfolio: VOL_ZSCORE) —
    vol_ma  = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std().replace(0, np.nan)
    df['vol_zscore'] = (df['volume'] - vol_ma) / vol_std

    # — Log-retornos y target 5 días adelante —
    df['log_ret']   = np.log(c / c.shift(1))
    df['fwd_ret_5'] = df['log_ret'].shift(-5).rolling(5).sum()

    return df


# ─────────────────────────────────────────────────────────────
#  3. MODELO 2 — MACD Strategy (AI.2-portfolio, copia exacta)
# ─────────────────────────────────────────────────────────────

def model2_signals(df: pd.DataFrame) -> pd.Series:
    """
    Señal binaria 0/1 del AI.2-portfolio.

    COMPRA cuando se cumplen las 4 condiciones simultáneamente:
      1. MACD histograma cruza de negativo → positivo
      2. MACD < 0  (momentum aún negativo, cruce temprano)
      3. Signal < 0 (confirmación)
      4. Precio > EMA200 (tendencia alcista de largo plazo)

    La salida es gestionada por el motor de backtesting con SL/TP.
    """
    sig = pd.Series(0, index=df.index, name='model2')
    hist = df['MACD_hist'].values
    macd = df['MACD'].values
    msig = df['MACD_signal'].values
    tup  = df['trend_up'].values

    for i in range(1, len(df)):
        if (hist[i-1] < 0 and hist[i] > 0
                and macd[i] < 0
                and msig[i]  < 0
                and tup[i]   == 1):
            sig.iloc[i] = 1

    return sig


# ─────────────────────────────────────────────────────────────
#  4. MODELO 1 — ML Strategy (AI-portfolio, versión simplificada)
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BB_width', 'BB_pos',
    'NATR', 'ROC_5', 'ROC_20',
    'vol_zscore',
    'price_vs_sma20', 'price_vs_sma50',
    'MOM_10', 'STOCH_K', 'STOCH_D',
    'trend_up'
]


def model1_signals(df: pd.DataFrame,
                   train_ratio: float = 0.60,
                   min_train: int = 252,
                   retrain_every: int = 60) -> pd.Series:
    """
    Señal de probabilidad [0, 1] del AI-portfolio (versión simplificada).

    Implementa los conceptos clave del AI-portfolio:
      - Target: ¿el retorno en los próximos 5 días será positivo?
      - Modelo: Random Forest (stand-in de LightGBM; mismos principios)
      - Walk-forward training: re-entrena cada `retrain_every` días
      - Purge gap: 5 días de separación entre train y test
        (evita contaminación por los retornos solapados)
      - StandardScaler antes de cada entrenamiento

    Devuelve probabilidad P(retorno_5d > 0) para cada día de test.
    """
    # Target binario: retorno 5 días adelante > 0
    target = (df['fwd_ret_5'] > 0).astype(int)

    # Índices válidos (sin NaN)
    valid_mask = df[FEATURE_COLS].notna().all(axis=1) & target.notna()
    valid_idx  = df.index[valid_mask]
    df_c  = df.loc[valid_idx, FEATURE_COLS].copy()
    tgt_c = target.loc[valid_idx].copy()

    n         = len(df_c)
    train_end = int(n * train_ratio)

    probs = pd.Series(0.5, index=df.index, name='model1')

    if train_end < min_train:
        print(f"  [WARN] Datos insuficientes para entrenar ML ({train_end} < {min_train})")
        return probs

    # Entrenamiento inicial
    model  = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        min_samples_leaf=15, random_state=42,
        n_jobs=-1
    )
    scaler = StandardScaler()

    X_init = df_c.iloc[:train_end].values
    y_init = tgt_c.iloc[:train_end].values
    scaler.fit(X_init)
    model.fit(scaler.transform(X_init), y_init)

    # Walk-forward prediction con re-entrenamiento periódico
    for i in range(train_end, n):
        # Re-entrena cada `retrain_every` días con todos los datos disponibles
        # (con purge gap de 5 días para evitar look-ahead)
        if (i - train_end) % retrain_every == 0 and i > train_end:
            X_tr = df_c.iloc[:i - 5].values
            y_tr = tgt_c.iloc[:i - 5].values
            if len(X_tr) >= min_train:
                scaler.fit(X_tr)
                model.fit(scaler.transform(X_tr), y_tr)

        row = df_c.iloc[i:i+1].values
        prob = model.predict_proba(scaler.transform(row))[0][1]
        probs.loc[valid_idx[i]] = prob

    return probs


# ─────────────────────────────────────────────────────────────
#  5. ENSEMBLE HÍBRIDO
# ─────────────────────────────────────────────────────────────

def hybrid_signals(m1_probs: pd.Series,
                   m2_binary: pd.Series,
                   w1: float = W_MODEL1,
                   w2: float = W_MODEL2) -> pd.Series:
    """
    Combina ambas señales con pesos ponderados.

    Model 1 ya devuelve probabilidades [0,1].
    Model 2 devuelve 0 ó 1, lo tratamos como una señal discreta.
    El score combinado supera 0.55 solo si ambos modelos confirman.
    """
    combined = w1 * m1_probs + w2 * m2_binary.astype(float)
    return pd.Series(combined.values, index=m1_probs.index, name='hybrid')


# ─────────────────────────────────────────────────────────────
#  6. MOTORES DE BACKTESTING
# ─────────────────────────────────────────────────────────────

def backtest_tp_sl(df: pd.DataFrame,
                   signals: pd.Series,
                   sl: float = STOP_LOSS_PCT,
                   tp: float = TAKE_PROFIT_PCT,
                   name: str = 'Strategy') -> pd.Series:
    """
    Backtest con Take Profit / Stop Loss (estilo AI.2-portfolio).

    Entra en posición cuando signals == 1.
    Cierra cuando:
      • precio ≥ entry * (1 + tp)  → ganancia
      • precio ≤ entry * (1 - sl)  → pérdida

    Usado para Model 2 (MACD) y Hybrid.
    """
    cash    = 1.0
    in_pos  = False
    entry_p = tgt_p = stp_p = 0.0
    vals = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        sig   = signals.iloc[i]

        if not in_pos:
            if sig == 1:
                in_pos  = True
                entry_p = price
                tgt_p   = price * (1 + tp)
                stp_p   = price * (1 - sl)
        else:
            if price >= tgt_p:
                cash   *= tgt_p / entry_p
                in_pos  = False
            elif price <= stp_p:
                cash   *= stp_p / entry_p
                in_pos  = False

        vals.append(cash * (price / entry_p) if in_pos else cash)

    return pd.Series(vals, index=df.index, name=name)


def backtest_regime(df: pd.DataFrame,
                    probs: pd.Series,
                    enter_thr: float = 0.55,
                    exit_thr:  float = 0.48,
                    name: str = 'Strategy') -> pd.Series:
    """
    Backtest de régimen basado en probabilidades ML (estilo AI-portfolio).

    Concepto: el modelo ML decide CUÁNDO estar en el mercado.
      • Entra en SPY cuando prob > enter_thr (régimen alcista)
      • Sale a cash cuando prob < exit_thr  (régimen bajista)
      • Mientras está invertido, captura el retorno diario de SPY

    Esto refleja la idea central del AI-portfolio: usar ML para
    predecir el régimen de mercado y filtrar períodos de riesgo.
    """
    prices = df['close'].values
    cash   = 1.0
    in_mkt = False
    entry_p = 0.0
    vals = []

    for i in range(len(df)):
        price = prices[i]
        prob  = probs.iloc[i]

        if not in_mkt:
            if prob > enter_thr:
                in_mkt  = True
                entry_p = price
        else:
            if prob < exit_thr:
                # Realizamos el retorno acumulado
                cash   *= price / entry_p
                in_mkt  = False

        vals.append(cash * (price / entry_p) if in_mkt else cash)

    return pd.Series(vals, index=df.index, name=name)


def backtest_hybrid(df: pd.DataFrame,
                    probs: pd.Series,
                    macd_sig: pd.Series,
                    ml_thr: float = 0.58,
                    exit_thr: float = 0.48,
                    sl: float = STOP_LOSS_PCT,
                    tp: float = TAKE_PROFIT_PCT,
                    name: str = 'Hybrid Ensemble') -> pd.Series:
    """
    Backtest Híbrido: combina régimen ML con disciplina MACD.

    ENTRA cuando se cumple CUALQUIERA de estas condiciones:
      A) Señal MACD (AI.2-portfolio) + ML confirma (prob > 0.50)
         → Usa TP/SL del Modelo 2

      B) ML muy confiado solo (prob > ml_thr) sin señal MACD
         → Usa lógica de régimen (sale cuando prob cae)

    Esto explota la fortaleza de cada modelo:
      • MACD: excelente para capturas precisas con bajo riesgo
      • ML: excelente para identificar tendencias largas
    """
    prices   = df['close'].values
    cash     = 1.0
    in_pos   = False
    mode_pos = None          # 'tp_sl' o 'regime'
    entry_p  = tgt_p = stp_p = 0.0
    vals = []

    for i in range(len(df)):
        price   = prices[i]
        prob    = probs.iloc[i]
        m2_fire = macd_sig.iloc[i] == 1

        if not in_pos:
            # Condición A: MACD + ML confirma
            if m2_fire and prob > 0.50:
                in_pos   = True
                mode_pos = 'tp_sl'
                entry_p  = price
                tgt_p    = price * (1 + tp)
                stp_p    = price * (1 - sl)
            # Condición B: ML solo, muy confiado
            elif prob > ml_thr:
                in_pos   = True
                mode_pos = 'regime'
                entry_p  = price
        else:
            if mode_pos == 'tp_sl':
                if price >= tgt_p:
                    cash   *= tgt_p / entry_p
                    in_pos  = False
                elif price <= stp_p:
                    cash   *= stp_p / entry_p
                    in_pos  = False
            else:  # regime
                if prob < exit_thr:
                    cash   *= price / entry_p
                    in_pos  = False

        vals.append(cash * (price / entry_p) if in_pos else cash)

    return pd.Series(vals, index=df.index, name=name)


def buyhold(df: pd.DataFrame) -> pd.Series:
    """Benchmark pasivo: compra SPY el primer día y mantiene."""
    norm = df['close'] / df['close'].iloc[0]
    return pd.Series(norm.values, index=df.index, name='SPY Buy & Hold')


# ─────────────────────────────────────────────────────────────
#  7. MÉTRICAS DE RENDIMIENTO
# ─────────────────────────────────────────────────────────────

def metrics(portfolio: pd.Series, rf: float = 0.04) -> dict:
    """
    Calcula las tres métricas clave de rendimiento:

    CAGR (Compound Annual Growth Rate):
      Tasa de crecimiento anual compuesta.
      CAGR = (Valor_Final / Valor_Inicial)^(1/años) - 1

    Sharpe Ratio:
      Rentabilidad ajustada al riesgo.
      Sharpe = (Retorno_Medio - Tasa_Libre_Riesgo) / Desviación * sqrt(252)
      > 1.0 es bueno, > 2.0 es excelente

    Max Drawdown:
      Mayor caída desde un máximo histórico hasta un mínimo posterior.
      Mide el peor escenario de pérdida.
    """
    v = np.array(portfolio.values, dtype=float)

    # CAGR
    n_years = len(v) / 252
    cagr    = (v[-1] / v[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe
    daily   = pd.Series(v).pct_change().dropna()
    daily_rf = rf / 252
    excess  = daily - daily_rf
    sharpe  = (excess.mean() / excess.std() * np.sqrt(252)
               if excess.std() > 0 else 0.0)

    # Max Drawdown
    cum_max = pd.Series(v).cummax()
    dd      = (pd.Series(v) - cum_max) / cum_max
    max_dd  = dd.min()

    return {
        'CAGR':                     f'{cagr*100:+.2f}%',
        'Sharpe Ratio':             f'{sharpe:.3f}',
        'Max Drawdown':             f'{max_dd*100:.2f}%',
        'Retorno Total':            f'{(v[-1]/v[0]-1)*100:+.1f}%',
        'Valor Final ($1→)':        f'${v[-1]:.3f}',
    }


# ─────────────────────────────────────────────────────────────
#  8. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────

def print_table(all_metrics: dict) -> None:
    """Imprime la tabla de comparación de los 4 modelos en consola."""
    COL = 22
    names = list(all_metrics.keys())
    metric_names = list(next(iter(all_metrics.values())).keys())

    sep = '=' * (28 + COL * len(names))
    print('\n' + sep)
    print('  TABLA COMPARATIVA FINAL — 4 MODELOS')
    print(sep)

    header = f"{'Métrica':<28}" + ''.join(f'{n:<{COL}}' for n in names)
    print(header)
    print('-' * len(sep))

    for m in metric_names:
        row = f'{m:<28}' + ''.join(f'{all_metrics[n][m]:<{COL}}' for n in names)
        print(row)

    print(sep)


# ─────────────────────────────────────────────────────────────
#  9. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

DARK_BG   = '#0f0f1a'
PANEL_BG  = '#16162a'
GRID_CLR  = '#2a2a4a'
TXT_CLR   = '#e8e8f0'

PALETTE = {
    'SPY Buy & Hold':  '#4ecdc4',
    'Model 1 (ML)':   '#ff6b6b',
    'Model 2 (MACD)': '#ffd93d',
    'Hybrid Ensemble':'#6bcb77',
}


def _style_ax(ax, title='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TXT_CLR, fontsize=10, fontweight='bold', pad=6)
    ax.set_ylabel(ylabel, color=TXT_CLR, fontsize=8)
    ax.tick_params(colors=TXT_CLR, labelsize=7)
    ax.grid(alpha=0.25, color=GRID_CLR, linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)


def plot_all(df, portfolios, sigs, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 15), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.45, wspace=0.30,
                            top=0.93, bottom=0.05)

    # ── Panel 1: Portfolios (ancho completo) ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for name, port in portfolios.items():
        lw = 2.5 if name == 'Hybrid Ensemble' else 1.5
        ax1.plot(port.index, port.values,
                 color=PALETTE[name], linewidth=lw, label=name)
    _style_ax(ax1,
              title='Comparación de Portfolios — Todos los Modelos vs SPY',
              ylabel='Valor ($1 invertido)')
    ax1.legend(loc='upper left',
               facecolor='#1a1a2e', labelcolor=TXT_CLR, fontsize=8)

    # ── Panel 2: Precio SPY + señales híbridas ────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df.index, df['close'], color='#4ecdc4', lw=1.2, label='SPY')
    ax2.plot(df.index, df['EMA200'], color='#ff6b6b',
             lw=1, ls='--', label='EMA 200')
    buy_mask = sigs['hybrid'] > ENTRY_THRESHOLD
    ax2.scatter(df.index[buy_mask], df['close'][buy_mask],
                color='#6bcb77', marker='^', s=18, zorder=5,
                label=f'Hybrid Buy (>{ENTRY_THRESHOLD})', alpha=0.8)
    _style_ax(ax2,
              title='SPY — Precio + Señales de Compra (Hybrid)',
              ylabel='Precio ($)')
    ax2.legend(facecolor='#1a1a2e', labelcolor=TXT_CLR, fontsize=7)

    # ── Panel 3: MACD histograma (AI.2-portfolio) ─────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df.index, df['MACD'],        color='#4ecdc4', lw=1, label='MACD')
    ax3.plot(df.index, df['MACD_signal'], color='#ff6b6b', lw=1, label='Signal')
    hist_colors = ['#6bcb77' if v >= 0 else '#ff6b6b'
                   for v in df['MACD_hist']]
    ax3.bar(df.index, df['MACD_hist'],
            color=hist_colors, alpha=0.6, width=1, label='Histograma')
    ax3.axhline(0, color=TXT_CLR, lw=0.5)
    _style_ax(ax3,
              title='MACD — Indicador Central (AI.2-portfolio)',
              ylabel='MACD')
    ax3.legend(facecolor='#1a1a2e', labelcolor=TXT_CLR, fontsize=7)

    # ── Panel 4: Probabilidad del ML (AI-portfolio) ───────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df.index, sigs['model1'],
             color='#ff6b6b', lw=1, label='P(retorno > 0)')
    ax4.axhline(0.50, color=TXT_CLR,    lw=0.8, ls='--', label='50%')
    ax4.axhline(0.55, color='#ffd93d',  lw=0.8, ls='--', label='Umbral 55%')
    ax4.fill_between(df.index, 0.50, sigs['model1'],
                     where=sigs['model1'] > 0.50,
                     color='#6bcb77', alpha=0.15)
    ax4.set_ylim(0, 1)
    _style_ax(ax4,
              title='Model 1 — Probabilidad ML (Random Forest)',
              ylabel='P(retorno_5d > 0)')
    ax4.legend(facecolor='#1a1a2e', labelcolor=TXT_CLR, fontsize=7)

    # ── Panel 5: Score Híbrido ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df.index, sigs['hybrid'],
             color='#6bcb77', lw=1, label='Score Híbrido')
    ax5.axhline(ENTRY_THRESHOLD, color='#ffd93d',
                lw=0.8, ls='--', label=f'Umbral {ENTRY_THRESHOLD}')
    ax5.fill_between(df.index, ENTRY_THRESHOLD, sigs['hybrid'],
                     where=sigs['hybrid'] > ENTRY_THRESHOLD,
                     color='#6bcb77', alpha=0.3, label='Señal Activa')
    ax5.set_ylim(0, 1)
    _style_ax(ax5,
              title=f'Ensemble Híbrido ({int(W_MODEL1*100)}% ML + {int(W_MODEL2*100)}% MACD)',
              ylabel='Score Combinado')
    ax5.legend(facecolor='#1a1a2e', labelcolor=TXT_CLR, fontsize=7)

    fig.suptitle(
        'Hybrid S&P 500 Investment Model — Análisis Completo',
        color=TXT_CLR, fontsize=15, fontweight='bold'
    )

    out_path = os.path.join(out_dir, 'hybrid_model_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Gráfico guardado → {out_path}')


# ─────────────────────────────────────────────────────────────
#  10. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print('  HYBRID S&P 500 INVESTMENT MODEL')
    print('  AI-portfolio  ×  AI.2-portfolio  →  Ensemble')
    print('=' * 62)

    # ── Paso 1: Cargar datos ──────────────────────────────────
    print('\n[1/5] Cargando datos de SPY...')
    df = load_spy_data(SPY_PATH)
    print(f'  {df["date"].iloc[0].date()}  →  {df["date"].iloc[-1].date()}'
          f'  ({len(df)} barras diarias)')
    df = df.set_index('date')

    # ── Paso 2: Calcular indicadores ──────────────────────────
    print('\n[2/5] Calculando indicadores técnicos...')
    df = compute_features(df)
    print(f'  Features: {len(FEATURE_COLS)} indicadores técnicos')

    # ── Paso 3: Generar señales ───────────────────────────────
    print('\n[3/5] Generando señales de cada modelo...')

    print('  • Model 2 (MACD — AI.2-portfolio)...')
    m2 = model2_signals(df)
    n_m2 = m2.sum()
    print(f'    Señales de compra detectadas: {n_m2}')

    print('  • Model 1 (ML — AI-portfolio, walk-forward RF)...')
    m1 = model1_signals(df, train_ratio=0.60, min_train=252, retrain_every=60)
    print(f'    Probabilidad media en test: {m1[m1 != 0.5].mean():.3f}')

    # Señal híbrida para el gráfico: ML prob + MACD confirmation
    print('  • Hybrid Ensemble...')
    hyb_score = W_MODEL1 * m1 + W_MODEL2 * m2.astype(float)
    n_hyb = (hyb_score > ENTRY_THRESHOLD).sum()
    # Condición B: ML muy confiado solo
    n_ml_alone = (m1 > 0.58).sum()
    print(f'    MACD+ML activos: {n_hyb} | ML-solo (>0.58): {n_ml_alone}')

    sigs = pd.DataFrame({'model1': m1, 'model2': m2.astype(float), 'hybrid': hyb_score},
                        index=df.index)

    # ── Paso 4: Backtesting ───────────────────────────────────
    print('\n[4/5] Ejecutando backtests...')

    port_spy  = buyhold(df)
    port_m1   = backtest_regime(df, m1, enter_thr=0.55, exit_thr=0.48,
                                name='Model 1 (ML)')
    port_m2   = backtest_tp_sl(df, m2, name='Model 2 (MACD)')
    port_hyb  = backtest_hybrid(df, m1, m2, ml_thr=0.58, exit_thr=0.48,
                                name='Hybrid Ensemble')

    portfolios = {
        'SPY Buy & Hold':  port_spy,
        'Model 1 (ML)':   port_m1,
        'Model 2 (MACD)': port_m2,
        'Hybrid Ensemble': port_hyb,
    }

    # ── Paso 5: Métricas y tabla ──────────────────────────────
    print('\n[5/5] Calculando métricas de rendimiento...')
    all_metrics = {name: metrics(port) for name, port in portfolios.items()}
    print_table(all_metrics)

    # ── Visualización ─────────────────────────────────────────
    print('\nGenerando visualizaciones...')
    plot_all(df, portfolios, sigs)

    print('\n✓ Modelo híbrido completado.')
    print(f'  → Gráfico: {OUTPUT_DIR}/hybrid_model_analysis.png')


if __name__ == '__main__':
    main()
