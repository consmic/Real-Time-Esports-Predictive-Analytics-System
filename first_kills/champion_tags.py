"""
Champion tags for draft-based feature engineering.

Defines early-game behavior tags for champions used in first-kills prediction.
"""

import pandas as pd

# Champion tags: early-game strength, scaling, engage, skirmish, poke
# Values: 0 = weak/none, 1 = strong/present
CHAMPION_TAGS = {
    # ADC
    "Draven": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Lucian": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Jinx": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Caitlyn": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Ezreal": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Varus": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Ashe": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 1},
    "Jhin": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Kai'Sa": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Vayne": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Sivir": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Tristana": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Miss Fortune": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Xayah": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Aphelios": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Zeri": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Samira": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Nilah": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    
    # Top
    "Renekton": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Darius": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Garen": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Jax": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Fiora": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Camille": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Irelia": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Riven": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Aatrox": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "K'Sante": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Sett": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Gnar": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 1},
    "Jayce": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Gangplank": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Yorick": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Nasus": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Tryndamere": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Yone": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Yasuo": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Malphite": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Ornn": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Sion": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Cho'Gath": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Maokai": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Poppy": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Kennen": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 1},
    "Vladimir": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Rumble": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 1},
    "Teemo": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 0, "poke": 1},
    "Quinn": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 1},
    "Vayne": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    
    # Jungle
    "Lee Sin": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Elise": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Rek'Sai": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Xin Zhao": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Vi": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Jarvan IV": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Graves": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Kindred": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Viego": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Diana": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Ekko": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Kha'Zix": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Rengar": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Evelynn": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Shaco": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Nocturne": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Hecarim": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Udyr": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Warwick": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Olaf": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Trundle": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Volibear": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Sejuani": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Zac": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Amumu": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Nunu & Willump": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Gragas": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Maokai": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Ivern": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Lillia": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Taliyah": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Nidalee": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 0, "poke": 1},
    "Bel'Veth": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Briar": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    
    # Mid
    "Zed": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Talon": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Yasuo": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Yone": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Akali": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Katarina": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "LeBlanc": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Fizz": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Kassadin": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Kassadin": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Vladimir": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Ryze": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Azir": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Corki": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Orianna": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Syndra": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Xerath": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Ziggs": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Vel'Koz": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Lux": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Ahri": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 1},
    "Neeko": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 1},
    "Annie": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Brand": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Malzahar": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Cassiopeia": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Karthus": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Viktor": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Aurelion Sol": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Taliyah": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Galio": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Pantheon": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Qiyana": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Irelia": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Sylas": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Vex": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Akshan": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 1},
    "Naafiri": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},
    "Hwei": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    
    # Support
    "Thresh": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Blitzcrank": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Nautilus": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Leona": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Alistar": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Rakan": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 1, "poke": 0},
    "Pyke": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Bard": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Braum": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Taric": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Shen": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Galio": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Poppy": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 0, "poke": 0},
    "Sett": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Pantheon": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},
    "Zilean": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Karma": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Lulu": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Janna": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Nami": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Soraka": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Sona": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Yuumi": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Senna": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Seraphine": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Miliao": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Renata Glasc": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 0},
    "Lux": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Xerath": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Vel'Koz": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Brand": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Zyra": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Heimerdinger": {"early": 1, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    "Shaco": {"early": 1, "scaling": 0, "engage": 0, "skirmish": 1, "poke": 0},
    "Tahm Kench": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Taric": {"early": 0, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Rell": {"early": 1, "scaling": 1, "engage": 1, "skirmish": 0, "poke": 0},
    "Milio": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 0, "poke": 1},
    
    # 2025/2026 New Champions
    "Yunara": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},  # ADC
    "Ambessa": {"early": 1, "scaling": 0, "engage": 1, "skirmish": 1, "poke": 0},  # Top/Jungle
    "Aurora": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},  # Mid/Top
    "Zaahen": {"early": 0, "scaling": 1, "engage": 0, "skirmish": 1, "poke": 0},  # Top
}


def get_champion_tags(champion_name: str) -> dict:
    """
    Get tags for a champion.
    
    Args:
        champion_name: Name of the champion (case-insensitive)
        
    Returns:
        Dictionary with tags (early, scaling, engage, skirmish, poke)
        Returns all zeros if champion not found
    """
    if not champion_name or pd.isna(champion_name):
        return {"early": 0, "scaling": 0, "engage": 0, "skirmish": 0, "poke": 0}
    
    # Try exact match first
    if champion_name in CHAMPION_TAGS:
        return CHAMPION_TAGS[champion_name].copy()
    
    # Try case-insensitive match
    champion_name_lower = champion_name.lower()
    for key, value in CHAMPION_TAGS.items():
        if key.lower() == champion_name_lower:
            return value.copy()
    
    # Not found, return zeros
    return {"early": 0, "scaling": 0, "engage": 0, "skirmish": 0, "poke": 0}


def compute_team_comp_features(picks: list) -> dict:
    """
    Compute team composition features from a list of champion picks.
    
    Args:
        picks: List of 5 champion names (can include None/NaN)
        
    Returns:
        Dictionary with comp features (sum_early, mean_early, sum_engage, etc.)
    """
    tags_list = [get_champion_tags(pick) for pick in picks if pick and pd.notna(pick)]
    
    if len(tags_list) == 0:
        return {
            "sum_early": 0, "mean_early": 0,
            "sum_scaling": 0, "mean_scaling": 0,
            "sum_engage": 0, "mean_engage": 0,
            "sum_skirmish": 0, "mean_skirmish": 0,
            "sum_poke": 0, "mean_poke": 0,
        }
    
    # Sum and mean for each tag
    result = {}
    for tag in ["early", "scaling", "engage", "skirmish", "poke"]:
        values = [t[tag] for t in tags_list]
        result[f"sum_{tag}"] = sum(values)
        result[f"mean_{tag}"] = sum(values) / len(tags_list) if len(tags_list) > 0 else 0
    
    return result


# Role-based champion tags with detailed early-game attributes
# Scale: 0-10 (5 = neutral, 10 = very strong)
CHAMPION_ROLE_TAGS = {
    # TOP LANE
    "Renekton": {
        "role": "TOP",
        "early_strength": 9,
        "skirmish": 7,
        "lane_prio": 8,
        "kill_pressure": 7,
        "scaling": 2,
    },
    "Jax": {
        "role": "TOP",
        "early_strength": 4,
        "skirmish": 8,
        "lane_prio": 5,
        "kill_pressure": 8,
        "scaling": 9,
    },
    "Darius": {
        "role": "TOP",
        "early_strength": 9,
        "skirmish": 8,
        "lane_prio": 7,
        "kill_pressure": 9,
        "scaling": 3,
    },
    "Garen": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 5,
        "lane_prio": 6,
        "kill_pressure": 6,
        "scaling": 4,
    },
    "Fiora": {
        "role": "TOP",
        "early_strength": 5,
        "skirmish": 9,
        "lane_prio": 6,
        "kill_pressure": 8,
        "scaling": 9,
    },
    "Camille": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 8,
        "lane_prio": 7,
        "kill_pressure": 8,
        "scaling": 8,
    },
    "Irelia": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 9,
        "lane_prio": 7,
        "kill_pressure": 8,
        "scaling": 7,
    },
    "Riven": {
        "role": "TOP",
        "early_strength": 8,
        "skirmish": 9,
        "lane_prio": 7,
        "kill_pressure": 9,
        "scaling": 7,
    },
    "Aatrox": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 8,
        "lane_prio": 7,
        "kill_pressure": 7,
        "scaling": 7,
    },
    "K'Sante": {
        "role": "TOP",
        "early_strength": 6,
        "skirmish": 7,
        "lane_prio": 6,
        "kill_pressure": 6,
        "scaling": 8,
    },
    "Sett": {
        "role": "TOP",
        "early_strength": 8,
        "skirmish": 8,
        "lane_prio": 7,
        "kill_pressure": 8,
        "scaling": 4,
    },
    "Gnar": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 6,
        "lane_prio": 8,
        "kill_pressure": 6,
        "scaling": 7,
    },
    "Jayce": {
        "role": "TOP",
        "early_strength": 8,
        "skirmish": 6,
        "lane_prio": 9,
        "kill_pressure": 7,
        "scaling": 6,
    },
    "Gangplank": {
        "role": "TOP",
        "early_strength": 4,
        "skirmish": 5,
        "lane_prio": 6,
        "kill_pressure": 4,
        "scaling": 9,
    },
    "Yone": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 9,
        "lane_prio": 6,
        "kill_pressure": 8,
        "scaling": 8,
    },
    "Yasuo": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 9,
        "lane_prio": 6,
        "kill_pressure": 8,
        "scaling": 8,
    },
    "Malphite": {
        "role": "TOP",
        "early_strength": 3,
        "skirmish": 4,
        "lane_prio": 4,
        "kill_pressure": 3,
        "scaling": 7,
    },
    "Ornn": {
        "role": "TOP",
        "early_strength": 4,
        "skirmish": 5,
        "lane_prio": 5,
        "kill_pressure": 4,
        "scaling": 8,
    },
    "Sion": {
        "role": "TOP",
        "early_strength": 5,
        "skirmish": 4,
        "lane_prio": 5,
        "kill_pressure": 5,
        "scaling": 7,
    },
    "Maokai": {
        "role": "TOP",
        "early_strength": 4,
        "skirmish": 4,
        "lane_prio": 5,
        "kill_pressure": 3,
        "scaling": 7,
    },
    "Poppy": {
        "role": "TOP",
        "early_strength": 6,
        "skirmish": 5,
        "lane_prio": 5,
        "kill_pressure": 5,
        "scaling": 5,
    },
    "Kennen": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 6,
        "lane_prio": 8,
        "kill_pressure": 7,
        "scaling": 7,
    },
    "Rumble": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 6,
        "lane_prio": 8,
        "kill_pressure": 7,
        "scaling": 6,
    },
    "Quinn": {
        "role": "TOP",
        "early_strength": 8,
        "skirmish": 7,
        "lane_prio": 9,
        "kill_pressure": 7,
        "scaling": 5,
    },
    "Tryndamere": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 8,
        "lane_prio": 6,
        "kill_pressure": 8,
        "scaling": 7,
    },
    
    # JUNGLE
    "Lee Sin": {
        "role": "JUNGLE",
        "early_strength": 10,
        "skirmish": 9,
        "gank_pressure": 10,
        "invade": 8,
        "scaling": 2,
    },
    "Viego": {
        "role": "JUNGLE",
        "early_strength": 6,
        "skirmish": 8,
        "gank_pressure": 7,
        "invade": 6,
        "scaling": 6,
    },
    "Elise": {
        "role": "JUNGLE",
        "early_strength": 9,
        "skirmish": 8,
        "gank_pressure": 9,
        "invade": 7,
        "scaling": 2,
    },
    "Rek'Sai": {
        "role": "JUNGLE",
        "early_strength": 9,
        "skirmish": 8,
        "gank_pressure": 9,
        "invade": 8,
        "scaling": 3,
    },
    "Xin Zhao": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 8,
        "gank_pressure": 8,
        "invade": 7,
        "scaling": 4,
    },
    "Vi": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 8,
        "gank_pressure": 9,
        "invade": 6,
        "scaling": 5,
    },
    "Jarvan IV": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 8,
        "gank_pressure": 9,
        "invade": 6,
        "scaling": 4,
    },
    "Graves": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 7,
        "gank_pressure": 6,
        "invade": 9,
        "scaling": 7,
    },
    "Kindred": {
        "role": "JUNGLE",
        "early_strength": 5,
        "skirmish": 7,
        "gank_pressure": 6,
        "invade": 7,
        "scaling": 9,
    },
    "Diana": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 9,
        "gank_pressure": 7,
        "invade": 6,
        "scaling": 7,
    },
    "Ekko": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 8,
        "gank_pressure": 7,
        "invade": 6,
        "scaling": 7,
    },
    "Kha'Zix": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 8,
        "gank_pressure": 7,
        "invade": 8,
        "scaling": 8,
    },
    "Rengar": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 8,
        "gank_pressure": 8,
        "invade": 7,
        "scaling": 7,
    },
    "Nocturne": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 7,
        "gank_pressure": 9,
        "invade": 5,
        "scaling": 5,
    },
    "Hecarim": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 8,
        "gank_pressure": 8,
        "invade": 5,
        "scaling": 7,
    },
    "Udyr": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 7,
        "gank_pressure": 7,
        "invade": 6,
        "scaling": 5,
    },
    "Warwick": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 8,
        "gank_pressure": 8,
        "invade": 6,
        "scaling": 4,
    },
    "Olaf": {
        "role": "JUNGLE",
        "early_strength": 9,
        "skirmish": 9,
        "gank_pressure": 7,
        "invade": 8,
        "scaling": 4,
    },
    "Trundle": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 7,
        "gank_pressure": 6,
        "invade": 7,
        "scaling": 5,
    },
    "Volibear": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 8,
        "gank_pressure": 8,
        "invade": 6,
        "scaling": 5,
    },
    "Sejuani": {
        "role": "JUNGLE",
        "early_strength": 4,
        "skirmish": 5,
        "gank_pressure": 7,
        "invade": 3,
        "scaling": 7,
    },
    "Zac": {
        "role": "JUNGLE",
        "early_strength": 4,
        "skirmish": 5,
        "gank_pressure": 8,
        "invade": 3,
        "scaling": 7,
    },
    "Amumu": {
        "role": "JUNGLE",
        "early_strength": 4,
        "skirmish": 4,
        "gank_pressure": 8,
        "invade": 3,
        "scaling": 6,
    },
    "Nunu & Willump": {
        "role": "JUNGLE",
        "early_strength": 6,
        "skirmish": 5,
        "gank_pressure": 8,
        "invade": 4,
        "scaling": 4,
    },
    "Gragas": {
        "role": "JUNGLE",
        "early_strength": 6,
        "skirmish": 6,
        "gank_pressure": 7,
        "invade": 5,
        "scaling": 6,
    },
    "Ivern": {
        "role": "JUNGLE",
        "early_strength": 3,
        "skirmish": 3,
        "gank_pressure": 6,
        "invade": 2,
        "scaling": 7,
    },
    "Lillia": {
        "role": "JUNGLE",
        "early_strength": 4,
        "skirmish": 5,
        "gank_pressure": 5,
        "invade": 4,
        "scaling": 8,
    },
    "Taliyah": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 6,
        "gank_pressure": 7,
        "invade": 5,
        "scaling": 7,
    },
    "Nidalee": {
        "role": "JUNGLE",
        "early_strength": 8,
        "skirmish": 6,
        "gank_pressure": 7,
        "invade": 9,
        "scaling": 5,
    },
    "Bel'Veth": {
        "role": "JUNGLE",
        "early_strength": 7,
        "skirmish": 8,
        "gank_pressure": 6,
        "invade": 7,
        "scaling": 8,
    },
    "Briar": {
        "role": "JUNGLE",
        "early_strength": 9,
        "skirmish": 9,
        "gank_pressure": 8,
        "invade": 7,
        "scaling": 6,
    },
    "Maokai": {
        "role": "JUNGLE",
        "early_strength": 4,
        "skirmish": 4,
        "gank_pressure": 7,
        "invade": 3,
        "scaling": 7,
    },
    
    # MID LANE
    "LeBlanc": {
        "role": "MID",
        "early_strength": 9,
        "lane_prio": 9,
        "roam": 8,
        "skirmish": 9,
        "kill_pressure": 9,
    },
    "Orianna": {
        "role": "MID",
        "early_strength": 3,
        "lane_prio": 5,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 2,
    },
    "Zed": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 7,
        "roam": 8,
        "skirmish": 9,
        "kill_pressure": 9,
    },
    "Talon": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 6,
        "roam": 9,
        "skirmish": 8,
        "kill_pressure": 8,
    },
    "Yasuo": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 6,
        "roam": 6,
        "skirmish": 9,
        "kill_pressure": 8,
    },
    "Yone": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 6,
        "roam": 6,
        "skirmish": 9,
        "kill_pressure": 8,
    },
    "Akali": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 6,
        "roam": 8,
        "skirmish": 9,
        "kill_pressure": 9,
    },
    "Katarina": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 5,
        "roam": 8,
        "skirmish": 9,
        "kill_pressure": 9,
    },
    "Fizz": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 6,
        "roam": 7,
        "skirmish": 9,
        "kill_pressure": 8,
    },
    "Kassadin": {
        "role": "MID",
        "early_strength": 2,
        "lane_prio": 2,
        "roam": 4,
        "skirmish": 3,
        "kill_pressure": 2,
    },
    "Vladimir": {
        "role": "MID",
        "early_strength": 3,
        "lane_prio": 4,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 3,
    },
    "Ryze": {
        "role": "MID",
        "early_strength": 3,
        "lane_prio": 4,
        "roam": 4,
        "skirmish": 4,
        "kill_pressure": 3,
    },
    "Azir": {
        "role": "MID",
        "early_strength": 4,
        "lane_prio": 6,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 3,
    },
    "Corki": {
        "role": "MID",
        "early_strength": 4,
        "lane_prio": 6,
        "roam": 4,
        "skirmish": 4,
        "kill_pressure": 4,
    },
    "Syndra": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 8,
        "roam": 5,
        "skirmish": 6,
        "kill_pressure": 7,
    },
    "Xerath": {
        "role": "MID",
        "early_strength": 5,
        "lane_prio": 8,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 5,
    },
    "Ziggs": {
        "role": "MID",
        "early_strength": 5,
        "lane_prio": 7,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 5,
    },
    "Vel'Koz": {
        "role": "MID",
        "early_strength": 5,
        "lane_prio": 7,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 5,
    },
    "Lux": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 8,
        "roam": 5,
        "skirmish": 5,
        "kill_pressure": 7,
    },
    "Ahri": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 7,
        "roam": 8,
        "skirmish": 7,
        "kill_pressure": 7,
    },
    "Neeko": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 7,
        "roam": 6,
        "skirmish": 6,
        "kill_pressure": 7,
    },
    "Annie": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 7,
        "roam": 6,
        "skirmish": 6,
        "kill_pressure": 8,
    },
    "Brand": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 7,
        "roam": 4,
        "skirmish": 5,
        "kill_pressure": 6,
    },
    "Malzahar": {
        "role": "MID",
        "early_strength": 4,
        "lane_prio": 5,
        "roam": 4,
        "skirmish": 4,
        "kill_pressure": 4,
    },
    "Cassiopeia": {
        "role": "MID",
        "early_strength": 4,
        "lane_prio": 5,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 4,
    },
    "Viktor": {
        "role": "MID",
        "early_strength": 4,
        "lane_prio": 6,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 4,
    },
    "Galio": {
        "role": "MID",
        "early_strength": 6,
        "lane_prio": 6,
        "roam": 9,
        "skirmish": 6,
        "kill_pressure": 5,
    },
    "Pantheon": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 7,
        "roam": 9,
        "skirmish": 8,
        "kill_pressure": 8,
    },
    "Qiyana": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 7,
        "roam": 8,
        "skirmish": 9,
        "kill_pressure": 9,
    },
    "Irelia": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 7,
        "roam": 6,
        "skirmish": 9,
        "kill_pressure": 8,
    },
    "Sylas": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 6,
        "roam": 7,
        "skirmish": 8,
        "kill_pressure": 7,
    },
    "Vex": {
        "role": "MID",
        "early_strength": 7,
        "lane_prio": 7,
        "roam": 6,
        "skirmish": 6,
        "kill_pressure": 7,
    },
    "Akshan": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 8,
        "roam": 8,
        "skirmish": 7,
        "kill_pressure": 8,
    },
    "Naafiri": {
        "role": "MID",
        "early_strength": 8,
        "lane_prio": 7,
        "skirmish": 8,
        "kill_pressure": 8,
        "roam": 6,
    },
    "Hwei": {
        "role": "MID",
        "early_strength": 5,
        "lane_prio": 7,
        "roam": 3,
        "skirmish": 4,
        "kill_pressure": 5,
    },
    
    # ADC
    "Draven": {
        "role": "ADC",
        "early_strength": 9,
        "2v2_killlane": 9,
        "scaling": 2,
    },
    "Lucian": {
        "role": "ADC",
        "early_strength": 8,
        "2v2_killlane": 8,
        "scaling": 4,
    },
    "Jinx": {
        "role": "ADC",
        "early_strength": 3,
        "2v2_killlane": 3,
        "scaling": 9,
    },
    "Caitlyn": {
        "role": "ADC",
        "early_strength": 8,
        "2v2_killlane": 7,
        "scaling": 7,
    },
    "Ezreal": {
        "role": "ADC",
        "early_strength": 5,
        "2v2_killlane": 4,
        "scaling": 8,
    },
    "Varus": {
        "role": "ADC",
        "early_strength": 7,
        "2v2_killlane": 6,
        "scaling": 7,
    },
    "Ashe": {
        "role": "ADC",
        "early_strength": 7,
        "2v2_killlane": 6,
        "scaling": 7,
    },
    "Jhin": {
        "role": "ADC",
        "early_strength": 8,
        "2v2_killlane": 7,
        "scaling": 7,
    },
    "Kai'Sa": {
        "role": "ADC",
        "early_strength": 5,
        "2v2_killlane": 7,
        "scaling": 9,
    },
    "Vayne": {
        "role": "ADC",
        "early_strength": 3,
        "2v2_killlane": 4,
        "scaling": 9,
    },
    "Sivir": {
        "role": "ADC",
        "early_strength": 4,
        "2v2_killlane": 4,
        "scaling": 8,
    },
    "Tristana": {
        "role": "ADC",
        "early_strength": 7,
        "2v2_killlane": 8,
        "scaling": 8,
    },
    "Miss Fortune": {
        "role": "ADC",
        "early_strength": 8,
        "2v2_killlane": 7,
        "scaling": 7,
    },
    "Xayah": {
        "role": "ADC",
        "early_strength": 5,
        "2v2_killlane": 7,
        "scaling": 8,
    },
    "Aphelios": {
        "role": "ADC",
        "early_strength": 4,
        "2v2_killlane": 5,
        "scaling": 9,
    },
    "Zeri": {
        "role": "ADC",
        "early_strength": 4,
        "2v2_killlane": 6,
        "scaling": 9,
    },
    "Samira": {
        "role": "ADC",
        "early_strength": 8,
        "2v2_killlane": 9,
        "scaling": 7,
    },
    "Nilah": {
        "role": "ADC",
        "early_strength": 5,
        "2v2_killlane": 8,
        "scaling": 8,
    },
    
    # SUPPORT
    "Thresh": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 9,
        "2v2_killlane": 8,
    },
    "Blitzcrank": {
        "role": "SUPPORT",
        "early_strength": 8,
        "engage": 9,
        "2v2_killlane": 9,
    },
    "Nautilus": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 9,
        "2v2_killlane": 8,
    },
    "Leona": {
        "role": "SUPPORT",
        "early_strength": 8,
        "engage": 10,
        "2v2_killlane": 9,
    },
    "Alistar": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 9,
        "2v2_killlane": 8,
    },
    "Rakan": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 9,
        "2v2_killlane": 8,
    },
    "Pyke": {
        "role": "SUPPORT",
        "early_strength": 9,
        "engage": 8,
        "2v2_killlane": 9,
    },
    "Bard": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 7,
        "2v2_killlane": 6,
    },
    "Braum": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 8,
        "2v2_killlane": 7,
    },
    "Taric": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 8,
        "2v2_killlane": 7,
    },
    "Shen": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 7,
        "2v2_killlane": 6,
    },
    "Galio": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 8,
        "2v2_killlane": 7,
    },
    "Poppy": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 7,
        "2v2_killlane": 6,
    },
    "Sett": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 8,
        "2v2_killlane": 8,
    },
    "Pantheon": {
        "role": "SUPPORT",
        "early_strength": 8,
        "engage": 8,
        "2v2_killlane": 9,
    },
    "Zilean": {
        "role": "SUPPORT",
        "early_strength": 3,
        "engage": 2,
        "2v2_killlane": 2,
    },
    "Karma": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 4,
        "2v2_killlane": 5,
    },
    "Lulu": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 3,
        "2v2_killlane": 3,
    },
    "Janna": {
        "role": "SUPPORT",
        "early_strength": 3,
        "engage": 2,
        "2v2_killlane": 2,
    },
    "Nami": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 6,
        "2v2_killlane": 6,
    },
    "Soraka": {
        "role": "SUPPORT",
        "early_strength": 3,
        "engage": 2,
        "2v2_killlane": 2,
    },
    "Sona": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 3,
        "2v2_killlane": 3,
    },
    "Yuumi": {
        "role": "SUPPORT",
        "early_strength": 2,
        "engage": 1,
        "2v2_killlane": 1,
    },
    "Senna": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 4,
        "2v2_killlane": 6,
    },
    "Seraphine": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 3,
        "2v2_killlane": 3,
    },
    "Milio": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 4,
        "2v2_killlane": 4,
    },
    "Renata Glasc": {
        "role": "SUPPORT",
        "early_strength": 4,
        "engage": 6,
        "2v2_killlane": 5,
    },
    "Lux": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 4,
        "2v2_killlane": 6,
    },
    "Xerath": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 3,
        "2v2_killlane": 4,
    },
    "Vel'Koz": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 3,
        "2v2_killlane": 4,
    },
    "Brand": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 4,
        "2v2_killlane": 6,
    },
    "Zyra": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 4,
        "2v2_killlane": 6,
    },
    "Heimerdinger": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 3,
        "2v2_killlane": 5,
    },
    "Shaco": {
        "role": "SUPPORT",
        "early_strength": 7,
        "engage": 5,
        "2v2_killlane": 7,
    },
    "Tahm Kench": {
        "role": "SUPPORT",
        "early_strength": 5,
        "engage": 7,
        "2v2_killlane": 6,
    },
    "Rell": {
        "role": "SUPPORT",
        "early_strength": 6,
        "engage": 9,
        "2v2_killlane": 8,
    },
    
    # 2025/2026 New Champions
    "Yunara": {
        "role": "ADC",
        "early_strength": 4,
        "skirmish": 7,
        "lane_prio": 5,
        "kill_pressure": 6,
        "2v2_killlane": 6,
        "scaling": 8,
    },
    "Ambessa": {
        "role": "TOP",
        "early_strength": 7,
        "skirmish": 8,
        "lane_prio": 7,
        "kill_pressure": 8,
        "engage": 7,
        "scaling": 4,
    },
    "Aurora": {
        "role": "MID",
        "early_strength": 5,
        "skirmish": 7,
        "lane_prio": 6,
        "kill_pressure": 6,
        "engage": 5,
        "scaling": 7,
    },
    "Zaahen": {
        "role": "TOP",
        "early_strength": 4,
        "skirmish": 6,
        "lane_prio": 5,
        "kill_pressure": 5,
        "engage": 4,
        "scaling": 8,
    },
}


def get_role_tag(champion_name: str, attribute: str, default: float = 5.0) -> float:
    """
    Get a role-based attribute for a champion.
    
    Args:
        champion_name: Name of the champion
        attribute: Attribute name (e.g., "early_strength", "lane_prio")
        default: Default value if champion or attribute not found (neutral = 5)
        
    Returns:
        Attribute value (float)
    """
    if not champion_name or pd.isna(champion_name):
        return default
    
    # Try exact match first
    if champion_name in CHAMPION_ROLE_TAGS:
        return CHAMPION_ROLE_TAGS[champion_name].get(attribute, default)
    
    # Try case-insensitive match
    champion_name_lower = champion_name.lower()
    for key, value in CHAMPION_ROLE_TAGS.items():
        if key.lower() == champion_name_lower:
            return value.get(attribute, default)
    
    # Not found, return default (neutral)
    return default

