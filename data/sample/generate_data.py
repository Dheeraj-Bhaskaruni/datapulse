"""Generate realistic synthetic sample data for DataPulse."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent

FIRST_NAMES = [
    "James", "LeBron", "Stephen", "Kevin", "Giannis", "Luka", "Jayson", "Nikola",
    "Joel", "Shai", "Anthony", "Damian", "Devin", "Ja", "Donovan", "Tyrese",
    "Jalen", "Paolo", "Victor", "Chet", "Trae", "Zion", "Brandon", "Darius",
    "Scottie", "Evan", "Cade", "Franz", "Desmond", "Alperen", "Marcus", "Tyler",
    "Mikal", "Keldon", "Jordan", "Cameron", "Anfernee", "DeAaron", "Jaren", "Lauri",
    "Bam", "Jimmy", "Pascal", "DeMar", "Karl-Anthony", "Rudy", "Myles", "Dejounte",
    "Chris", "Russell",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Wilson", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez",
    "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen",
    "Young", "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson",
    "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker",
    "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed",
    "Cook", "Morgan",
]

TEAMS = [
    "Lakers", "Celtics", "Warriors", "Bucks", "Nuggets", "76ers", "Heat", "Suns",
    "Mavericks", "Grizzlies", "Cavaliers", "Kings", "Knicks", "Nets", "Hawks",
    "Pelicans", "Timberwolves", "Thunder", "Clippers", "Raptors",
]

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
SPORTS = ["NBA", "NFL", "MLB", "NHL"]
CONTEST_TYPES = ["head2head", "tournament", "50-50", "multiplier", "satellite"]


def generate_players(n: int = 500) -> pd.DataFrame:
    """Generate realistic player data."""
    names = []
    used: set = set()
    for _ in range(n):
        while True:
            name = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"
            if name not in used:
                used.add(name)
                names.append(name)
                break

    positions = np.random.choice(POSITIONS, n)
    teams = np.random.choice(TEAMS, n)

    # Position-based stat distributions
    points_base = {'PG': 18, 'SG': 20, 'SF': 17, 'PF': 16, 'C': 15}
    assists_base = {'PG': 7, 'SG': 4, 'SF': 3.5, 'PF': 3, 'C': 2.5}
    rebounds_base = {'PG': 4, 'SG': 4.5, 'SF': 6, 'PF': 8, 'C': 10}

    points = np.array([points_base[p] + np.random.normal(0, 5) for p in positions]).clip(3, 38).round(1)
    assists = np.array([assists_base[p] + np.random.normal(0, 2) for p in positions]).clip(0.5, 13).round(1)
    rebounds = np.array([rebounds_base[p] + np.random.normal(0, 2.5) for p in positions]).clip(1, 15).round(1)

    games = np.random.randint(10, 83, n)
    steals = np.random.uniform(0.3, 2.2, n).round(1)
    blocks = np.random.uniform(0.1, 2.5, n).round(1)
    turnovers = np.random.uniform(0.8, 4.5, n).round(1)
    fg_pct = np.random.uniform(0.38, 0.62, n).round(3)

    # Fantasy points calculation (DraftKings-style)
    fantasy_points = (
        points * 1.0 + assists * 1.5 + rebounds * 1.25 +
        steals * 2.0 + blocks * 2.0 - turnovers * 0.5
    ).round(1)

    # Salary correlates with fantasy points
    salary = (fantasy_points * 150 + np.random.normal(0, 500, n)).clip(3500, 12000).astype(int)

    consistency = np.random.uniform(0.3, 0.95, n).round(3)
    injury_status = np.random.choice(
        ['healthy', 'questionable', 'out', 'day-to-day'],
        n, p=[0.75, 0.12, 0.05, 0.08],
    )

    return pd.DataFrame({
        'player_id': range(1, n + 1),
        'name': names,
        'team': teams,
        'position': positions,
        'games_played': games,
        'points_avg': points,
        'assists_avg': assists,
        'rebounds_avg': rebounds,
        'steals_avg': steals,
        'blocks_avg': blocks,
        'turnovers_avg': turnovers,
        'fg_pct': fg_pct,
        'fantasy_points': fantasy_points,
        'salary': salary,
        'consistency_score': consistency,
        'injury_status': injury_status,
    })


def generate_contests(n: int = 1000) -> pd.DataFrame:
    """Generate realistic contest data."""
    types = np.random.choice(CONTEST_TYPES, n, p=[0.2, 0.35, 0.25, 0.12, 0.08])
    sports = np.random.choice(SPORTS, n, p=[0.4, 0.35, 0.15, 0.1])

    entry_fees = np.random.choice(
        [0.25, 1, 3, 5, 10, 20, 25, 50, 100, 250, 500],
        n, p=[0.05, 0.15, 0.15, 0.15, 0.15, 0.1, 0.08, 0.08, 0.05, 0.03, 0.01],
    )

    # Determine max entries based on contest type
    max_entries_raw = []
    for t in types:
        if t == 'head2head':
            max_entries_raw.append(2)
        elif t == 'tournament':
            max_entries_raw.append(int(np.random.choice([50, 100, 500, 1000, 5000, 10000])))
        elif t == '50-50':
            max_entries_raw.append(int(np.random.choice([10, 20, 50, 100])))
        elif t == 'multiplier':
            max_entries_raw.append(int(np.random.choice([20, 50, 100])))
        else:
            max_entries_raw.append(int(np.random.choice([10, 25, 50])))

    max_entries = np.array(max_entries_raw)
    current_entries = (max_entries * np.random.uniform(0.5, 1.0, n)).astype(int).clip(1)
    prize_pools = (entry_fees * max_entries * np.random.uniform(0.85, 0.95, n)).round(2)

    dates = pd.date_range('2023-01-01', '2024-12-31', periods=n)
    status = np.random.choice(
        ['completed', 'live', 'upcoming', 'cancelled'],
        n, p=[0.65, 0.1, 0.2, 0.05],
    )

    return pd.DataFrame({
        'contest_id': range(1, n + 1),
        'sport': sports,
        'contest_type': types,
        'entry_fee': entry_fees,
        'prize_pool': prize_pools,
        'max_entries': max_entries,
        'current_entries': current_entries,
        'start_time': dates,
        'status': status,
    })


def generate_user_entries(n: int = 2000) -> pd.DataFrame:
    """Generate realistic user entry data."""
    user_ids = np.random.randint(1, 301, n)
    contest_ids = np.random.randint(1, 1001, n)

    # Generate lineups as JSON lists
    lineups = [
        json.dumps(sorted(np.random.choice(range(1, 501), 8, replace=False).tolist()))
        for _ in range(n)
    ]

    total_scores = np.random.normal(180, 40, n).clip(50, 350).round(1)
    payouts = np.where(
        np.random.random(n) > 0.55,
        0,
        total_scores * np.random.uniform(0.5, 3, n),
    ).round(2)

    entry_times = pd.date_range('2023-01-01', '2024-12-31', periods=n)
    ranks = np.random.randint(1, 100, n)

    return pd.DataFrame({
        'entry_id': range(1, n + 1),
        'user_id': user_ids,
        'contest_id': contest_ids,
        'lineup': lineups,
        'total_score': total_scores,
        'payout': payouts,
        'entry_time': entry_times,
        'rank': ranks,
    })


def generate_market_odds(n: int = 1500) -> pd.DataFrame:
    """Generate realistic market odds data."""
    sports = np.random.choice(SPORTS, n, p=[0.4, 0.35, 0.15, 0.1])
    market_types = np.random.choice(
        ['spread', 'moneyline', 'over_under', 'player_prop'],
        n, p=[0.3, 0.25, 0.25, 0.2],
    )

    selections = [f"Selection_{i}" for i in range(1, n + 1)]

    opening_lines = np.random.uniform(-10, 10, n).round(1)
    line_shift = np.random.normal(0, 1.5, n)
    closing_lines = (opening_lines + line_shift).round(1)

    # American odds centered around -110
    opening_odds = np.random.choice(
        [-250, -200, -175, -150, -130, -120, -115, -110, -105,
         100, 105, 110, 115, 120, 130, 150, 175, 200, 250],
        n,
    )
    odds_shift = np.random.choice([-10, -5, 0, 0, 0, 5, 10], n)
    closing_odds = opening_odds + odds_shift

    results = np.random.randint(0, 2, n)
    timestamps = pd.date_range('2023-01-01', '2024-12-31', periods=n)

    return pd.DataFrame({
        'event_id': range(1, n + 1),
        'sport': sports,
        'market_type': market_types,
        'selection': selections,
        'opening_line': opening_lines,
        'closing_line': closing_lines,
        'opening_odds': opening_odds,
        'closing_odds': closing_odds,
        'result': results,
        'timestamp': timestamps,
    })


def generate_user_profiles(n: int = 300) -> pd.DataFrame:
    """Generate realistic user profile data."""
    usernames = [f"user_{i:04d}" for i in range(1, n + 1)]

    join_dates = pd.date_range('2020-01-01', '2024-06-01', periods=n)
    last_active = pd.date_range('2024-01-01', '2024-12-31', periods=n)

    total_entries = np.random.lognormal(5, 1.5, n).astype(int).clip(1, 50000)
    total_contests = (total_entries * np.random.uniform(0.7, 1.0, n)).astype(int)

    # Win rate follows beta distribution (slightly below 50% for most)
    win_rate = np.random.beta(4, 5, n).round(3)

    avg_entry_fee = np.random.lognormal(2, 1, n).clip(0.25, 500).round(2)
    total_wagered = (total_entries * avg_entry_fee * np.random.uniform(0.8, 1.2, n)).round(2)

    # Most users lose money, some win
    roi = np.random.normal(-0.05, 0.15, n)
    total_won = (total_wagered * (1 + roi)).clip(0).round(2)
    net_profit = (total_won - total_wagered).round(2)

    # Risk score correlates with win rate and volume
    risk_score = (
        win_rate * 40 + np.log1p(total_wagered) / 12 * 30 + np.random.normal(0, 10, n)
    ).clip(0, 100).round(1)

    account_tiers = np.where(
        total_wagered > 50000, 'platinum',
        np.where(total_wagered > 10000, 'gold',
        np.where(total_wagered > 1000, 'silver', 'bronze')),
    )

    return pd.DataFrame({
        'user_id': range(1, n + 1),
        'username': usernames,
        'join_date': join_dates,
        'total_entries': total_entries,
        'total_contests': total_contests,
        'win_rate': win_rate,
        'avg_entry_fee': avg_entry_fee,
        'total_wagered': total_wagered,
        'total_won': total_won,
        'net_profit': net_profit,
        'risk_score': risk_score,
        'last_active': last_active,
        'account_tier': account_tiers,
    })


def main():
    """Generate all sample datasets."""
    print("Generating sample data for DataPulse...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = {
        'players': (generate_players, {}),
        'contests': (generate_contests, {}),
        'user_entries': (generate_user_entries, {}),
        'market_odds': (generate_market_odds, {}),
        'user_profiles': (generate_user_profiles, {}),
    }

    for name, (gen_func, kwargs) in generators.items():
        df = gen_func(**kwargs)
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  [OK] {name}.csv: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nAll sample data generated successfully!")


if __name__ == "__main__":
    main()
