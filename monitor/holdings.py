# monitor/holdings.py
# Persists actual holdings and trade log to data/holdings.json

import json
import os
from datetime import datetime, timezone
from pathlib import Path

HOLDINGS_FILE = "data/holdings.json"

def _default():
    return {"weights": {}, "initialized": False, "last_updated": None, "trade_log": []}

def load() -> dict:
    if not Path(HOLDINGS_FILE).exists():
        return _default()
    try:
        with open(HOLDINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return _default()

def save(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_weights() -> dict:
    return load().get("weights", {})

def is_initialized() -> bool:
    return load().get("initialized", False)

def initialize_to_target(target_weights: dict):
    """Set actual holdings = target weights (day 1 setup)."""
    data = load()
    data["weights"] = {k: float(v) for k, v in target_weights.items()}
    data["initialized"] = True
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["trade_log"].append({
        "date": datetime.now(timezone.utc).isoformat(),
        "action": "INITIALIZED",
        "note": "Set actual = target weights",
        "weights_snapshot": data["weights"].copy(),
    })
    save(data)

def log_trade(ticker: str, action: str, old_weight: float, new_weight: float, note: str = ""):
    data = load()
    data["weights"][ticker] = new_weight
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["trade_log"].append({
        "date": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": action,
        "old_weight": old_weight,
        "new_weight": new_weight,
        "note": note,
    })
    save(data)

def update_weights(new_weights: dict, note: str = "Manual update"):
    data = load()
    old = data["weights"].copy()
    data["weights"] = {k: float(v) for k, v in new_weights.items()}
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["trade_log"].append({
        "date": datetime.now(timezone.utc).isoformat(),
        "action": "MANUAL_UPDATE",
        "note": note,
        "old_weights": old,
        "new_weights": data["weights"].copy(),
    })
    save(data)

def get_trade_log() -> list:
    return load().get("trade_log", [])
