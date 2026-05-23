import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Moving Averages in Stock Trading",
        "category": "technical_analysis",
        "content": """
Moving Averages are one of the most important technical indicators used by traders to identify trends.

Definition:
A moving average is the average price of a stock over a specific time period. As new prices come in, 
the oldest prices are dropped from the calculation.

Types of Moving Averages:

1. Simple Moving Average (SMA):
   - Calculation: Sum of closing prices / Number of periods
   - Example: 50-day SMA = Sum of last 50 closing prices / 50
   - Characteristics: Equal weight to all periods, slower to respond to price changes

2. Exponential Moving Average (EMA):
   - Calculation: More recent prices given higher weight
   - Characteristics: Faster to respond to price changes, more sensitive
   - Formula: EMA = Price(today) × K + EMA(yesterday) × (1-K), where K = 2/(N+1)

Trading Applications:

1. Trend Identification:
   - Price above MA = Uptrend
   - Price below MA = Downtrend
   - MA sloping upward = Bullish
   - MA sloping downward = Bearish

2. Support and Resistance:
   - Price bounces off moving averages frequently
   - 50-day and 200-day MAs are major support/resistance levels

3. Golden Cross / Death Cross:
   - Golden Cross: 50-day MA crosses above 200-day MA = BUY signal
   - Death Cross: 50-day MA crosses below 200-day MA = SELL signal

4. Mean Reversion:
   - When price moves far from MA, it tends to revert back
   - This creates trading opportunities

Common Timeframes:
- 10-day MA: Short-term traders (scalpers)
- 50-day MA: Intermediate traders (weeks to months)
- 200-day MA: Long-term investors (months to years)

Best Practices:
- Use multiple MAs for confirmation (e.g., 20, 50, 200)
- Combine with other indicators (RSI, MACD) for better signals
- Don't rely on MA alone; consider price action, volume, and overall market conditions
"""
    },

    {
        "id": "doc_002",
        "title": "Support and Resistance Levels",
        "category": "technical_analysis",
        "content": """
Support and Resistance are fundamental concepts in technical analysis used to identify key price levels 
where stocks often reverse or consolidate.

Support Levels:
- Price level below current price where buyers typically emerge
- Prevents price from falling further
- Represents demand zone
- Formed when price stops declining and reverses upward multiple times
- Example: If AAPL bounces off $150 three times, $150 is support

Resistance Levels:
- Price level above current price where sellers typically emerge
- Prevents price from rising further
- Represents supply zone
- Formed when price stops rising and reverses downward multiple times
- Example: If AAPL fails to break above $180 three times, $180 is resistance

Characteristics of Strong Levels:

1. Multiple Touch Points:
   - More touches = stronger level
   - Level tested 3+ times = very reliable
   - Single touch = weak signal

2. Time Frame Matters:
   - Levels that held for months are stronger than weekly levels
   - Combine different timeframes for confluence

3. Volume at Levels:
   - High volume touches = stronger support/resistance
   - Low volume touches = may break through easily

Trading Strategies Using Support/Resistance:

1. Bounce Trading:
   - Buy near support with stop loss below
   - Sell near resistance with take profit above
   - Risk/Reward ratio typically 1:2 or 1:3

2. Breakout Trading:
   - Price breaks above resistance = Buy signal
   - Price breaks below support = Sell signal
   - Requires volume confirmation for validity

3. Range Trading:
   - Buy support, sell resistance in sideways market
   - Exit strategy when support/resistance breaks

Real Example (Stock XYZ):
- Support: $95, $92, $88
- Resistance: $105, $110, $115
- Current Price: $100
- Trade Setup: Buy at $95 (support) with stop at $92, target $105 (resistance)
- Risk: $3 per share, Reward: $5 per share (Risk/Reward = 1:1.67)

Key Rules:
1. Never buy at resistance or sell at support
2. Always use stop losses below support or above resistance
3. Stronger levels have been tested multiple times
4. Volume confirmation is crucial for breakouts
5. Support becomes resistance when broken (and vice versa)
"""
    },

    {
        "id": "doc_003",
        "title": "RSI (Relative Strength Index) Indicator",
        "category": "technical_analysis",
        "content": """
The Relative Strength Index (RSI) is a momentum oscillator that measures the magnitude of recent price 
changes to evaluate overbought or oversold conditions.

Basic Information:
- Range: 0 to 100 (bounded oscillator)
- Creator: J. Welles Wilder Jr. (1978)
- Standard Period: 14 (can be adjusted: 7 for faster, 21 for slower)
- Formula: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss

Interpretation:

RSI Levels:
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- RSI 30-70: Neutral zone
- RSI > 50: Bullish momentum
- RSI < 50: Bearish momentum

Trading Signals:

1. Overbought/Oversold (Most Common):
   - RSI > 70: Stock may be overextended upward
     → Look for reversal or pullback
     → Sell signal (especially if price near resistance)

   - RSI < 30: Stock may be overextended downward
     → Look for reversal or bounce
     → Buy signal (especially if price near support)

2. Divergence (More Advanced):
   - Bullish Divergence: Price makes lower low but RSI makes higher low → Potential BUY
   - Bearish Divergence: Price makes higher high but RSI makes lower high → Potential SELL
   - Often signals trend reversal before price confirms

3. Trend Confirmation:
   - In strong uptrend: RSI stays above 50, often above 70
   - In strong downtrend: RSI stays below 50, often below 30
   - RSI crossing 50 can signal trend change

Best Practices:

1. Don't Trade RSI Alone:
   - Combine with price action, support/resistance, volume
   - Example: RSI < 30 + Price at support + High volume = Strong BUY

2. Timeframe Selection:
   - 1-hour chart: For day traders (faster signals)
   - 4-hour chart: For swing traders
   - Daily chart: For position traders
   - Weekly chart: Long-term trend analysis

3. Adjust Period for Market Conditions:
   - RSI(7): More sensitive, more false signals
   - RSI(14): Standard, good balance
   - RSI(21): Smoother, filters noise

4. In Range-Bound Markets:
   - Overbought/oversold signals work best
   - Buy RSI < 30, Sell RSI > 70

5. In Trending Markets:
   - Divergence signals are more reliable
   - High RSI in uptrend is NORMAL, not sell signal

Real Example:
AAPL Daily Chart Analysis:
- Current RSI: 32 (Oversold)
- Current Price: $145 (At support level $145)
- Volume: Higher than average
- Signal: BUY with stop loss at $143 (below support)

Common Mistakes:
- Selling when RSI > 70 in strong uptrend (misses gains)
- Buying when RSI < 30 without checking support/resistance
- Using too-short period (7) causing too many false signals
- Ignoring divergence at extreme levels
"""
    },

    {
        "id": "doc_004",
        "title": "Margin Trading Explained",
        "category": "trading_strategies",
        "content": """
Margin trading is when you borrow money from your broker to purchase securities with less capital upfront. 
It amplifies both gains and losses.

Key Concepts:

Initial Margin:
- Minimum amount of your own money you must deposit to open a margin account
- Typically 50% for stocks (required by regulation)
- Example: To buy $10,000 worth of stock, you deposit $5,000, broker lends $5,000

Maintenance Margin:
- Minimum equity you must maintain in account
- Typically 25% for stocks
- If equity falls below this, you get a margin call
- Formula: Maintenance Margin = (Loan Amount) / (Current Stock Value)

Margin Call:
- Broker demands you deposit more cash or sell securities
- Happens when account equity falls below maintenance margin
- Forces you to realize losses
- Example: You borrowed $5,000 to buy stock now worth $4,000. If margin = 25%, 
  you need $4,000 × 0.25 = $1,000 equity. You only have $500, so margin call triggers.

Pros of Margin Trading:

1. Increased Purchasing Power:
   - Control $10,000 with $5,000 (2x leverage)
   - Amplifies gains
   - Example: Stock rises 10%, you gain $1,000 on $5,000 = 20% return

2. Short Selling:
   - Borrow stock to sell it now, buy it back later
   - Profit if price falls
   - Example: Borrow 100 shares of XYZ at $50, sell for $5,000
     → Stock drops to $40, buy back for $4,000
     → Profit: $1,000 (minus interest and fees)

Cons and Risks:

1. Amplified Losses:
   - Example: Stock falls 10%
   - With $5,000 margin: You lose $1,000 on $5,000 = 20% loss
   - Worse: Stock could fall 50%, wiping out your entire equity and still owing broker

2. Interest Costs:
   - Broker charges interest on borrowed money
   - Typical rates: 5-12% annually
   - Reduces profitability

3. Margin Calls:
   - Forced to sell at worst time (market panic)
   - Realizes losses, locks in bad trades
   - Creates emotional pressure

4. Risk of Total Loss:
   - In theory, stock could go to zero
   - You still owe borrowed amount
   - Could lose more than initial investment

5. Liquidation Risk:
   - Broker can force-sell your positions
   - May sell best performers first
   - No control over execution

Regulations and Rules:

1. Reg T (Regulation T by Federal Reserve):
   - Sets initial margin at 50% for stocks
   - Broker can set higher requirements

2. Maintenance Margin:
   - FINRA: 25% minimum (but broker can require higher)
   - Many brokers: 30-40% to be safer

3. Uptick Rule (for short selling):
   - Can't short on downtick (prevents manipulation)
   - Must wait for uptick or zero-plus tick

Safe Margin Trading Practices:

1. Conservative Position Sizing:
   - Use only 2:1 leverage (borrow equal to deposit)
   - Never go to maximum allowed leverage
   - Example: $10,000 account, only use $5,000 margin

2. Always Use Stop Losses:
   - Protect against margin calls
   - Set stop loss to limit loss to 2% of account
   - Example: $10,000 account, stop loss at $200 loss

3. Monitor Positions Closely:
   - Check daily at minimum
   - Margin calls can happen quickly in volatile markets
   - Have exit plan before entering trade

4. Keep Cash Reserve:
   - Always keep 50% of margin requirement in cash
   - Cushion against margin call
   - Avoid being forced to liquidate

5. Understand Your Broker's Policies:
   - Margin interest rates vary
   - Maintenance requirements differ
   - Forced liquidation procedures

When NOT to Use Margin:
- Beginners (lack of experience with volatility)
- Volatile stocks (tech, biotech, penny stocks)
- During market uncertainty
- If you can't monitor positions regularly
- If you don't have emergency cash reserve

Real Scenario:
Account Balance: $10,000
Margin Requirement: 50% (you can borrow $10,000 total)
You deposit: $10,000, borrow: $10,000
Buy: $20,000 worth of stock (2:1 leverage)

Scenario A - Price rises 10%:
- Stock now worth: $22,000
- You owe: $10,000
- Your equity: $12,000 (20% gain on $10,000)

Scenario B - Price falls 10%:
- Stock now worth: $18,000
- You owe: $10,000
- Your equity: $8,000 (20% loss on $10,000)
- Still above 25% maintenance, no margin call

Scenario C - Price falls 30%:
- Stock now worth: $14,000
- You owe: $10,000
- Your equity: $4,000 (maintenance = $14,000 × 0.25 = $3,500)
- Still okay, but close to margin call

Scenario D - Price falls 40%:
- Stock now worth: $12,000
- You owe: $10,000
- Your equity: $2,000 (maintenance needed = $12,000 × 0.25 = $3,000)
- MARGIN CALL! Forced to deposit $1,000 or sell securities
"""
    },

    {
        "id": "doc_005",
        "title": "Options Trading Basics",
        "category": "derivatives",
        "content": """
Options are derivative contracts that give the buyer the right (not obligation) to buy or sell 
a stock at a specific price on or before a specific date.

Key Terminology:

Call Option:
- Right to BUY stock at strike price
- Buyer profits if stock price RISES
- Example: AAPL Call, Strike $150, Premium $5
  → Right to buy AAPL at $150
  → If AAPL rises to $160, profit = $160 - $150 - $5 (premium) = $5

Put Option:
- Right to SELL stock at strike price
- Buyer profits if stock price FALLS
- Example: AAPL Put, Strike $150, Premium $5
  → Right to sell AAPL at $150
  → If AAPL falls to $140, profit = $150 - $140 - $5 (premium) = $5

Strike Price:
- Price at which option contract can be exercised
- Different strikes available: In-the-Money (ITM), At-The-Money (ATM), Out-of-The-Money (OTM)

Expiration Date:
- Date when option contract expires and becomes worthless if not exercised
- Common expirations: Weekly, Monthly (3rd Friday), Quarterly
- After expiration: Option has ZERO value

Premium:
- Price paid for the option
- Represents the cost of the right
- Paid upfront, non-refundable

Greeks (Option Pricing Factors):

1. Delta (Δ):
   - How much option price moves with $1 stock move
   - Call Delta: 0 to +1 (positive)
   - Put Delta: -1 to 0 (negative)
   - ATM option ≈ 0.50 delta
   - Deep ITM option ≈ 1.0 delta
   - Deep OTM option ≈ 0.0 delta

2. Gamma (Γ):
   - Rate of delta change
   - Tells how delta will accelerate
   - High gamma = delta changes rapidly (near expiration, ATM)
   - Low gamma = delta stable (far expiration, deep ITM/OTM)

3. Theta (Θ):
   - Time decay (how much value option loses daily)
   - Always negative for option buyers
   - Positive for option sellers
   - Accelerates as expiration approaches
   - Example: Theta = -0.05, option loses $0.05/day

4. Vega (ν):
   - Volatility risk
   - How option price changes with implied volatility change
   - High volatility = higher option prices
   - Before earnings: high vega
   - After earnings: vega collapse

5. Rho (ρ):
   - Interest rate sensitivity
   - Less relevant for most traders

Basic Strategies:

1. Long Call (Bullish):
   - Buy call option
   - Profit: Unlimited (stock price can go infinitely high)
   - Loss: Limited to premium paid
   - Break-even: Strike + Premium
   - Best for: Stock expected to rise significantly

2. Long Put (Bearish):
   - Buy put option
   - Profit: Limited to (Strike - Premium)
   - Loss: Limited to premium paid
   - Break-even: Strike - Premium
   - Best for: Stock expected to fall significantly

3. Covered Call (Income):
   - Own stock + Sell call option
   - Premium: Income you keep
   - Called away: If stock rises above strike, stock gets bought
   - Benefit: Generate income on stock you hold
   - Drawback: Caps upside gain

4. Protective Put (Insurance):
   - Own stock + Buy put option
   - Put acts as insurance against downside
   - Cost: Premium paid
   - Benefit: Can sell stock at strike if price falls
   - Best for: Protecting gains in volatile stock

5. Spreads (Limited Risk):
   - Buy one option, sell another
   - Examples: Bull Call Spread, Bear Call Spread
   - Reduced cost and risk
   - Also reduced profit potential

In-The-Money (ITM) vs Out-of-The-Money (OTM):

Call Options:
- ITM: Stock price > Strike price (has intrinsic value)
- OTM: Stock price < Strike price (only time value)
- ATM: Stock price ≈ Strike price

Put Options:
- ITM: Stock price < Strike price (has intrinsic value)
- OTM: Stock price > Strike price (only time value)
- ATM: Stock price ≈ Strike price

Intrinsic vs Time Value:
- Intrinsic Value: Actual value if exercised today
- Time Value: Premium above intrinsic (value of time remaining)
- Total Option Price = Intrinsic + Time Value

Example:
AAPL stock $160, Call Strike $150:
- Intrinsic: $10 (could exercise and have $10)
- If option premium is $12:
  - Time value: $2 (market betting on more upside)

As expiration approaches:
- Time value → $0
- Option price → Intrinsic value

When to Exercise:
- Calls: Usually don't exercise, sell instead
- Puts: Exercise to capture stock at strike price
- Exception: Before dividend (exercise call to get dividend)

Risks in Options:

1. Time Decay (Theta):
   - OTM options lose value daily
   - Accelerates near expiration
   - Can lose 50% value with no stock price change

2. Volatility Collapse:
   - High volatility before earnings
   - After earnings, vega crush kills option values
   - Call bought before earnings often loses money

3. Assignment Risk:
   - Seller can force exercise on you
   - Covered call assigned: Stock sold away
   - Short put assigned: Forced to buy stock

4. Liquidity:
   - Far OTM options hard to exit
   - Wide bid-ask spreads
   - May not get filled at desired price

Real Example:
Today's Setup:
- AAPL trading at $150
- 30 days to expiration
- You expect AAPL to hit $160

Option Trade:
- Buy Call: Strike $150, Premium $3
- Cost: $300 (1 contract = 100 shares)
- Break-even: $153

Outcomes:
1. AAPL rises to $165:
   - Sell call for $15 (intrinsic)
   - Profit: $15 - $3 = $12 × 100 = $1,200

2. AAPL stays at $150:
   - Call expires worthless
   - Loss: $3 × 100 = $300 (100% loss of premium)

3. AAPL falls to $140:
   - Call expires worthless
   - Loss: $3 × 100 = $300 (same as #2)

Key Takeaway:
- For $300 at risk, controlled exposure to stock price movement
- Much cheaper than buying 100 shares ($15,000)
- Can lose 100% of premium but can't lose more
"""
    }
]


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def index_documents():
    """Main function to index all documents into ChromaDB."""

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Get or create collection
    collection = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Track chunks for logging
    total_chunks = 0

    # Process each document
    for doc in DOCUMENTS:
        doc_id = doc["id"]
        title = doc["title"]
        category = doc["category"]
        content = doc["content"]

        print(f"Processing: {title}")

        # Split content into chunks
        chunks = chunk_text(content, chunk_size=500, overlap=50)

        # Embed and index each chunk
        for chunk_idx, chunk_content in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            embedding = model.encode(chunk_content)
            collection.add(
                ids=[chunk_id],
                documents=[chunk_content],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "title": title,
                    "category": category,
                    "chunk_text": chunk_content[:100] + "..."
                }]
            )

            total_chunks += 1

        print(f"  ✓ Indexed {len(chunks)} chunks")

    print(f"\n✅ Total chunks indexed: {total_chunks}")
    print(f"📁 Saved to: ./chroma_db")

    return collection


def test_retrieval(query: str):
    """Test RAG retrieval with a sample query."""

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode query
    query_emb = model.encode([query])[0]

    # Retrieve
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Query: '{query}'")
    print("-" * 80)

    for i, (doc, meta, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        score = 1 - distance
        print(f"\nResult {i + 1} (Score: {score:.3f}):")
        print(f"  Document: {meta['title']} (Category: {meta['category']})")
        print(f"  Preview: {doc[:150]}...")

    return results


if __name__ == "__main__":
    # Step 1: Index all documents
    print("🚀 Starting document indexing...\n")
    collection = index_documents()

    # Step 2: Test retrieval with sample queries
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)

    test_queries = [
        "What is moving average and how to use it",
        "How does RSI indicator work",
        "Can you explain options trading",
        "What is margin trading risk",
        "Support and resistance levels"
    ]

    for query in test_queries:
        test_retrieval(query)
        print("\n" + "-" * 80)