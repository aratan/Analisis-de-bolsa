import yfinance as yf
import pandas as pd
import numpy as np
import gradio as gr
from openai import OpenAI

# Configurar Ollama como cliente OpenAI
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Obtiene los datos históricos de un activo financiero usando yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data.reset_index()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(data):
    """
    Calcula indicadores técnicos avanzados como SMA, RSI, MACD y rendimientos.
    """
    if data.empty:
        return data

    # Medias móviles simples
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Rendimientos diarios
    data['Daily_Return'] = data['Close'].pct_change()

    return data


def query_ollama(prompt):
    """
    Envía un prompt a un modelo LLM Ollama y obtiene una respuesta detallada.
    """
    try:
        print("ejecutando llamada al modelo")
        res = openai.chat.completions.create(
            model='tulu3',
            messages=[
                {'role': 'system',
                 'content': 'You are a highly skilled financial analyst providing actionable trading signals. Be direct and concise. Focus on buy/sell opportunities with specific price targets and dates. Provide a confidence level (0-100%) for each recommendation. Do not provide general information about the company or its financials. ONLY provide specific and actionable trading advice. Follow the specified format for your response.'},
                {'role': 'user', 'content': f'{prompt}'},
            ]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error querying Ollama: {e}"


def dataframe_to_markdown(df):
    """Converts a Pandas DataFrame to Markdown format."""
    md = df.to_markdown(index=False)
    return md


def analyze_stock(ticker):
    """
    Analiza un activo financiero, calcula indicadores técnicos y consulta un modelo LLM para obtener recomendaciones de trading.
    """
    # Paso 1: Obtener datos
    data = get_stock_data(ticker, period="1y", interval="1d")  # Increased period for more data
    if data.empty:
        return "No se pudieron obtener datos para el activo. Verifica el símbolo o la conexión a internet."

    # Paso 2: Calcular indicadores técnicos
    data = calculate_technical_indicators(data)

    # Paso 3: Preparar el prompt
    prompt = f"El activo {ticker} tiene los siguientes datos recientes:\n"
    columns = ['Date', 'Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line', 'Daily_Return']
    # Convert DataFrame to Markdown
    markdown_data = dataframe_to_markdown(data[columns].tail(30))
    prompt += markdown_data

    # Instrucciones claras para el modelo - VERY IMPORTANT!
    prompt += f"""
    \n\nBased ONLY on the provided data, provide specific, actionable trading recommendations.

    Focus on short-term trading opportunities (within the next week or two).

    Be extremely direct and concise. Do not provide background information or general analysis.

    Provide the following information in the specified format:

    *   **Trend:** [Overall trend (e.g., Bullish, Bearish, Sideways)]
    *   **Buy Signal:**
        *   **Date:** [Date to buy (YYYY-MM-DD)]
        *   **Price:** [Price to buy at]
        *   **Confidence:** [Confidence level (0-100%)]
    *   **Sell Signal:**
        *   **Date:** [Date to sell (YYYY-MM-DD)]
        *   **Price:** [Price to sell at]
        *   **Confidence:** [Confidence level (0-100%)]
    *   **Stop-Loss:** [Stop-loss price]

    If there are NO clear buy or sell opportunities, state that directly: "No clear trading signals at this time."
    """

    # Paso 4: Consultar el modelo LLM
    llm_response = query_ollama(prompt)

    # Devolver resultados
    return f"**Prompt enviado al modelo:**\n\n{prompt}\n\n**Respuesta del modelo:**\n\n{llm_response}"


# Interfaz Gradio
iface = gr.Interface(
    fn=analyze_stock,
    inputs=gr.Textbox(label="Introduce el símbolo del activo (ejemplo: AAPL)"),
    outputs=gr.Markdown(label="Resultados del análisis")  # Changed to gr.Markdown
)
iface.launch()
