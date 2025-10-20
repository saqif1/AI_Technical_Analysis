import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import openai

# ----------------------------
# Page Configuration (Enable Wide Mode)
# ----------------------------
st.set_page_config(
    page_title="AI Stock Technical Analysis",
    page_icon="📈",
    layout="wide"  # 👈 This makes the main area wide
)

model = "deepseek/deepseek-v3.2-exp"

# ----------------------------
# Initialize session state
# ----------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.analysis_text = ""
    st.session_state.ticker = "SPY"
    st.session_state.years_back = 3
    st.session_state.df = pd.DataFrame()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker (e.g., SPY, AAPL, BTC-USD)", value=st.session_state.ticker)
    years_back = st.number_input("Years of Historical Data", min_value=1, max_value=10, value=st.session_state.years_back)
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    
    # Generate button in sidebar
    st.divider()
    generate_clicked = st.button("🔄 Generate Analysis", use_container_width=True)

# ----------------------------
# Main Header
# ----------------------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1>📈 AI-Powered Stock Technical Analysis Dashboard</h1>
        <p style="font-size: 1.2em; color: #555;">
            Real-time technical insights powered by AI and market data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()


# Fetch data button
if generate_clicked:
    if not openrouter_key.strip():
        st.error("❌ Please enter your OpenRouter API key.")
    elif not ticker.strip():
        st.error("❌ Please enter a valid stock ticker.")
    else:
        with st.spinner("Fetching stock data and analyzing..."):
            try:
                # Date range
                end_date = dt.datetime.today()
                start_date = end_date - dt.timedelta(days=years_back * 365)

                # Download data
                df = yf.download(ticker, start=start_date, end=end_date)
                if df.empty:
                    st.error(f"No data found for ticker: {ticker}. Please check the symbol.")
                    st.stop()

                # Initialize OpenRouter client
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",  # No trailing spaces!
                    api_key=openrouter_key.strip()
                )

                # System prompt (your guide)
                system_prompt = """
You're a technical analysis pro with domain expertise in

below:

A Comprehensive Guide to Technical Analysis

This guide explains the core concepts of technical analysis,

its advantages and disadvantages, and various methods and patterns used by

traders, all described without visual aids.

I. Introduction to Technical Analysis

Technical analysis is a trading discipline that uses

historical price data to forecast future price movements. It is one of three

main "stylistic species" of traders, alongside fundamental and

quantitative traders. A key benefit of technical analysis is that it can be

combined with fundamental analysis to create a more robust market view. For

example, a trader might first develop a fundamental view based on supply and

demand and then use technical analysis to refine entry and exit points.

Advantages of Technical Analysis:

• Data Availability: Price data is often more readily

available and of higher quality than fundamental data.

• Clarity: It provides clear, actionable signals for when to

enter or exit a trade.

• Universality: The principles can be applied across

different markets and time frames.

• Incorporates Fundamentals: Technical analysis implicitly

considers all known fundamental information because that information is

reflected in the price.

Disadvantages of Technical Analysis:

• Self-Fulfilling Prophecy: Its effectiveness can sometimes

depend on a large number of market participants believing in and acting on the

same signals.

• Ambiguity: Patterns can be imperfect, and trend lines are

often inexact, leading to subjective interpretations.

• False Signals: It can generate signals that do not result

in the expected price movement.

A pragmatic approach is often best, where a trader uses

technical analysis as a tool but remains aware of its limitations.

II. The Core Concept: Trends

The foundation of technical analysis is the concept of a

trend, which is the general direction in which a market's price is moving.

• Definition of a Trend: A trend is established by a series

of price movements. An uptrend is characterized by a series of higher highs and

higher lows. A downtrend is characterized by a series of lower lows and lower

highs.

• Drawing Trend Lines: Trend lines are drawn to connect the

lows of an uptrend or the highs of a downtrend to visualize the trend's

trajectory. However, these lines can be inexact and subject to interpretation.

• Multiple Time Frames: Trends exist simultaneously across

different time frames, often described as a "fractal nature":

    ◦ Macro Trend: The

long-term trend, lasting months or years.

    ◦ Intermediate

Trend: A medium-term trend that occurs within the macro trend, lasting weeks or

months.

    ◦ Micro Trend: The

short-term trend, lasting days or weeks.

• End of a Trend: A trend is considered to be over when this

pattern is broken. For an uptrend, this occurs when the price makes a lower

low. For a downtrend, it happens when the price creates a higher high.

III. Support and Resistance

Support and resistance are key price levels where the

momentum of a trend is likely to pause or reverse.

• Support: A price level where buying interest is strong

enough to overcome selling pressure, causing a downtrend to pause or reverse.

It often occurs at a previous low point.

• Resistance: A price level where selling pressure is strong

enough to overcome buying interest, causing an uptrend to pause or reverse. It

often occurs at a previous high point.

• Congestion Levels: These are price areas where significant

trading activity has occurred in the past, often acting as strong zones of

support or resistance.

• Role Reversal: Once a resistance level is decisively

broken, it often becomes a new support level. Conversely, when a support level

is broken, it can turn into a resistance level.

IV. Technical Measurement Techniques

Traders use several techniques to measure price movements

and identify potential trading opportunities.

• Moving Averages: A moving average is a continuously

calculated average price over a specific number of periods (e.g., 50 days or

200 days). It helps to smooth out price fluctuations and identify the

underlying trend direction. A common strategy involves observing when a

shorter-term moving average crosses above or below a longer-term one.

• Bollinger Bands: These consist of a moving average plus

two bands plotted at a set number of standard deviations above and below it.

    ◦ The bands widen

during periods of high volatility and narrow during periods of low volatility.

    ◦ Prices are

considered high when they touch the upper band and low when they touch the

lower band.

• Fibonacci Retracements: This technique identifies

potential support or resistance levels based on the idea that after a

significant price move, the price will retrace a certain percentage of that

move before continuing in the original direction. These retracement levels are

based on specific mathematical ratios.

• Pattern-Implied Objectives: Certain chart patterns suggest

a potential price target. For instance, after a price breaks out of a

consolidation pattern, the size of the prior price range can be used to project

how far the price might travel.

• Volume: The number of units traded in a period can be a

confirmation indicator. A price move accompanied by high volume is generally

considered more significant than a move with low volume.

V. Common Technical Patterns

Technical analysis identifies recurring patterns in price

movements that can signal either a continuation of the current trend or a

reversal.

Continuation Patterns (Suggest the trend will resume):

• Bull Flag / Bear Flag: A brief period of consolidation in

the opposite direction of a strong trend, resembling a flag on a pole. A

breakout from this flag pattern signals the trend is likely to continue.

• Triangle Continuation Patterns:

    ◦ Flat Ascending

Triangle: Occurs in an uptrend when there is a flat resistance level and a

rising support trend line. A break above resistance signals continuation.

    ◦ Flat Descending

Triangle: Occurs in a downtrend with a flat support level and a falling

resistance trend line. A break below support signals continuation.

• Rectangle Consolidation: Price moves sideways between

parallel support and resistance levels before eventually breaking out in the

direction of the original trend.

Reversal Patterns (Suggest the trend is about to change):

• Head and Shoulders: A pattern typically seen at market

tops, consisting of three peaks: a central, higher peak (the head) flanked by

two lower peaks (the shoulders). A break below the "neckline" (the

support level connecting the lows of the pattern) signals a potential trend

reversal to the downside.

• Rounding Bottom: A gradual, bowl-shaped turn from a

downtrend to an uptrend, indicating a slow shift in market sentiment from

selling to buying.

It is important to note that sometimes no discernible

pattern exists, and in such cases, technical analysis offers no prediction.

 

Perform Technical Analysis given the raw csv data, so you

have to do calculations on the back-end using this data yourself



Please wait while i pass you the csv
                """.strip()

                # Send to OpenRouter
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Stock Technical Analysis Dashboard"
                    },
                    model=model,  # Free & available model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Perform comprehensive technical analysis on this stock:\n\n{df.to_string()}"}
                    ],
                    max_tokens=2000  # Reasonable limit
                )

                # Save to session state
                st.session_state.analysis_text = response.choices[0].message.content
                st.session_state.analysis_done = True
                st.session_state.ticker = ticker
                st.session_state.years_back = years_back
                st.session_state.df = df

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)  # Optional: show full traceback during debugging

# ----------------------------
# Display Results (if available)
# ----------------------------
if st.session_state.analysis_done:
    st.subheader("📊 AI Technical Analysis")
    st.markdown(st.session_state.analysis_text)

    # Download button (now safe – doesn't trigger re-run of analysis)
    report = f"""AI Technical Analysis Report
Generated on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {st.session_state.ticker}
Period: {st.session_state.years_back} years

{'='*60}

{st.session_state.analysis_text}
"""

    st.download_button(
        label="📥 Download Report (TXT)",
        data=report,
        file_name=f"{st.session_state.ticker}_technical_analysis_{dt.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("### Connect with Me!")
st.markdown("""
<a href="https://www.linkedin.com/in/saqif-juhaimee-17322a119/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" style="vertical-align: middle;">
    Saqif Juhaimee
</a>
""", unsafe_allow_html=True)