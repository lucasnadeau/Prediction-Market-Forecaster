# build_dataset.py
import os
import json
import pandas as pd
from dotenv import load_dotenv
from dome_api_sdk import DomeClient
import numpy as np
import math

import duckdb
import time


load_dotenv()
dome_api_key = os.getenv("DOME_API_KEY")
if not dome_api_key:
    raise RuntimeError("Set DOME_API_KEY in your environment first.")

dome = DomeClient({
    "api_key": dome_api_key,
    "timeout": 120.0,
})

OUT_DIR = "data/silver_features"
DB_PATH = "data/duckdb/pmml.duckdb"

MAX_MARKETS = 300      
PAGE_LIMIT = 25
WRITE_EVERY = 25       
SLEEP = 1.05

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def main():
    conn = duckdb.connect(DB_PATH)
    
    print("\n\nFetching markets with pagination...\n\n")
    suitable_markets = []
    offset = 0
    
    while len(suitable_markets) < MAX_MARKETS:
        print(f"Fetching page at offset {offset}...")
        resp = dome.polymarket.markets.get_markets({
            "status": "closed",
            "limit": PAGE_LIMIT,
            "offset": offset,
        })
        time.sleep(SLEEP)
        markets = resp.markets
        
        if not markets:
            print("No more markets available.")
            break
        
        for cand in markets:
            if cand.winning_side and cand.side_a and cand.side_b:
                a = (cand.side_a.label or "").lower()
                b = (cand.side_b.label or "").lower()
                if (a, b) in [("yes", "no"), ("no", "yes")]:
                    suitable_markets.append(cand)
                    if len(suitable_markets) >= MAX_MARKETS:
                        break
        
        print(f"Found {len(suitable_markets)}/{MAX_MARKETS} suitable markets so far...")
        offset += PAGE_LIMIT
    
    if not suitable_markets:
        raise RuntimeError("No suitable closed Yes/No markets found.")
    
    print(f"\n\nFound {len(suitable_markets)} suitable markets (target was {MAX_MARKETS})\n\n")

    all_dfs = []
    file_counter = 0
    for idx, m in enumerate(suitable_markets):
        market_start_time = time.time()
        print(f"\n\nProcessing market {idx+1}/{len(suitable_markets)}: {m.market_slug}\n\n")
        
        label = (m.winning_side.label or "").strip().lower()
        if label == 'yes':
            y = 1
        else:
            y = 0

        if m.end_time is None:
            print(f"Market {m.market_slug} has no end time, skipping...")
            continue
        end_time = int(m.end_time)
        start_time = end_time - 60 * 60 * 24 * 90 

        print("\n\nFetching candlesticks...\n\n")
        try:
            candles = dome.polymarket.markets.get_candlesticks({
                "condition_id": m.condition_id,
                "start_time": start_time,
                "end_time": end_time,
                "interval": 1440, 
            })
            time.sleep(SLEEP)
        except ValueError as e:
            market_elapsed = time.time() - market_start_time
            error_str = str(e).lower()
            if "504" in error_str or "gateway timeout" in error_str or "timeout" in error_str:
                print(f"\n\nServer timeout fetching candlesticks for {m.market_slug}, skipping... (took {market_elapsed:.2f}s)\n\n")
            else:
                print(f"\n\nError fetching candlesticks for {m.market_slug}: {e}, skipping... (took {market_elapsed:.2f}s)\n\n")
            continue
        except Exception as e:
            market_elapsed = time.time() - market_start_time
            print(f"\n\nUnexpected error fetching candlesticks for {m.market_slug}: {e}, skipping... (took {market_elapsed:.2f}s)\n\n")
            continue

        rows = []
        for stream in candles.candlesticks:
            token_meta = stream[-1]
            token_id = token_meta.token_id
            for c in stream[:-1]:
                d = vars(c)
                d["token_id"] = token_id
                rows.append(d)

        if not rows:
            market_elapsed = time.time() - market_start_time
            print(f"No candlestick data for {m.market_slug}, skipping... (took {market_elapsed:.2f}s)")
            continue

        df = pd.DataFrame(rows)

        price_df = pd.json_normalize(df["price"]).add_prefix("price.")
        ask_df   = pd.json_normalize(df["yes_ask"]).add_prefix("yes_ask.")
        bid_df   = pd.json_normalize(df["yes_bid"]).add_prefix("yes_bid.")

        df = pd.concat([df.drop(columns=["price", "yes_ask", "yes_bid"]), price_df, ask_df, bid_df], axis=1)

        for col in ["price.close_dollars", "yes_ask.close_dollars", "yes_bid.close_dollars",
                "yes_ask.open_dollars",  "yes_bid.open_dollars"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        

        df["p_mkt"] = (df["yes_bid.close_dollars"] + df["yes_ask.close_dollars"]) / 2.0
        df["p_mkt"] = df["p_mkt"].fillna(df["price.close_dollars"])
        df = df.dropna(subset=["p_mkt"])

        df["spread"] = (df["yes_ask.close_dollars"] - df["yes_bid.close_dollars"]).clip(lower=0.0)

        df["y_final"] = y
        df["market_slug"] = m.market_slug
        df["condition_id"] = m.condition_id
        
        df["market_end_time"] = int(m.end_time)
        m_start_time_attr = getattr(m, 'start_time', None)
        df["market_start_time"] = int(m_start_time_attr) if m_start_time_attr else None
        df["market_category"] = getattr(m, 'category', None) or getattr(m, 'group_item_title', None) or None
        df["market_volume_total"] = getattr(m, 'volume', None) or getattr(m, 'total_volume', None) or None
        df["market_liquidity"] = getattr(m, 'liquidity', None) or None

        if (df["p_mkt"] < 0.0).any():
            print("p_mkt is less than 0.0:")
        if (df["p_mkt"] > 1.0).any():
            print("p_mkt is greater than 1.0:")

        df["p_mkt"] = df["p_mkt"].clip(0.0, 1.0)

        df["time_to_close_sec"] = int(m.end_time) - df["end_period_ts"]

        df = df.sort_values("end_period_ts")
        df["ret_1"] = df["p_mkt"].diff()
        df["vol_7"] = df["ret_1"].rolling(7).std()

        df["had_trade"] = (df["volume"] > 0).astype(int)

        all_dfs.append(df)
        
        market_elapsed = time.time() - market_start_time
        print(f"Processed {m.market_slug} in {market_elapsed:.2f}s ({len(df)} rows)")
        
        if (idx + 1) % WRITE_EVERY == 0:
            batch_df = pd.concat(all_dfs, ignore_index=True)
            parquet_path = os.path.join(OUT_DIR, f"features_{file_counter:04d}.parquet")
            conn.register("batch_df", batch_df)
            conn.execute(f"COPY batch_df TO '{parquet_path}' (FORMAT PARQUET)")
            print(f"\n\nWrote {len(batch_df)} rows to {parquet_path}\n\n")
            all_dfs = []
            file_counter += 1

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        parquet_path = os.path.join(OUT_DIR, f"features_{file_counter:04d}.parquet")
        conn.register("final_df", final_df)
        conn.execute(f"COPY final_df TO '{parquet_path}' (FORMAT PARQUET)")
        print(f"\n\nWrote {len(final_df)} rows to {parquet_path}\n\n")
        file_counter += 1
    

    conn.execute(f"""CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{OUT_DIR}/*.parquet');""")
    print("\n\nCreated DuckDB view: features\n\n")
    conn.close()
    print(f"\n\nWrote {file_counter} parquet file(s) to {OUT_DIR}\n\n")




if __name__ == "__main__":
    main()
