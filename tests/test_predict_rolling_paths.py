import pandas as pd

import first_kills.predict as predict_mod


def _base_games(gameid):
    return pd.DataFrame(
        {
            "gameid": [gameid],
            "date": [pd.Timestamp("2026-01-01")],
            "blue_teamname": ["Blue"],
            "red_teamname": ["Red"],
        }
    )


def _base_teams(gameid):
    return pd.DataFrame(
        {
            "gameid": [gameid, gameid],
            "teamid": [f"{gameid}_b", f"{gameid}_r"],
            "side": ["Blue", "Red"],
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
        }
    )


def test_predict_from_raw_data_uses_combined_team_df_for_historical(monkeypatch):
    new_raw = pd.DataFrame({"src": ["new"]})
    hist_raw = pd.DataFrame({"src": ["hist"]})
    new_team = _base_teams("new_game")
    hist_team = _base_teams("hist_game")
    new_games = _base_games("new_game")
    hist_games = _base_games("hist_game")

    def fake_load_raw_data(path):
        return hist_raw if path == "hist.csv" else new_raw

    def fake_build_team_level(df_raw):
        return hist_team if df_raw is hist_raw else new_team

    def fake_build_game_level(team_df):
        return hist_games if team_df is hist_team else new_games

    rolling_calls = []

    def fake_add_team_rolling_stats(games_df, team_df, window=10):
        rolling_calls.append((games_df.copy(), team_df.copy(), window))
        out = games_df.copy()
        out["roll3_killsat10_diff"] = 0.0
        return out

    def fake_predict_first_kills_for_games(games_df, models_dir="output_first_kills", rolling_window=10, team_df=None):
        return pd.DataFrame({"gameid": games_df["gameid"].tolist()})

    monkeypatch.setattr(predict_mod, "load_raw_data", fake_load_raw_data)
    monkeypatch.setattr(predict_mod, "build_team_level", fake_build_team_level)
    monkeypatch.setattr(predict_mod, "build_game_level", fake_build_game_level)
    monkeypatch.setattr(predict_mod, "add_team_rolling_stats", fake_add_team_rolling_stats)
    monkeypatch.setattr(predict_mod, "predict_first_kills_for_games", fake_predict_first_kills_for_games)

    result = predict_mod.predict_from_raw_data(
        raw_data_path="new.csv",
        historical_data_path="hist.csv",
    )

    assert len(rolling_calls) == 1
    _, combined_team_df, _ = rolling_calls[0]
    assert set(combined_team_df["gameid"].unique()) == {"hist_game", "new_game"}
    assert result["gameid"].tolist() == ["new_game"]


def test_predict_from_raw_data_passes_current_team_df_without_historical(monkeypatch):
    new_raw = pd.DataFrame({"src": ["new"]})
    new_team = _base_teams("new_game")
    new_games = _base_games("new_game")

    def fake_load_raw_data(path):
        return new_raw

    def fake_build_team_level(df_raw):
        return new_team

    def fake_build_game_level(team_df):
        return new_games

    rolling_calls = []

    def fake_add_team_rolling_stats(games_df, team_df, window=10):
        rolling_calls.append((games_df.copy(), team_df.copy(), window))
        out = games_df.copy()
        out["roll3_killsat10_diff"] = 0.0
        return out

    def fake_predict_first_kills_for_games(games_df, models_dir="output_first_kills", rolling_window=10, team_df=None):
        return pd.DataFrame({"gameid": games_df["gameid"].tolist()})

    monkeypatch.setattr(predict_mod, "load_raw_data", fake_load_raw_data)
    monkeypatch.setattr(predict_mod, "build_team_level", fake_build_team_level)
    monkeypatch.setattr(predict_mod, "build_game_level", fake_build_game_level)
    monkeypatch.setattr(predict_mod, "add_team_rolling_stats", fake_add_team_rolling_stats)
    monkeypatch.setattr(predict_mod, "predict_first_kills_for_games", fake_predict_first_kills_for_games)

    result = predict_mod.predict_from_raw_data(raw_data_path="new.csv")

    assert len(rolling_calls) == 1
    _, passed_team_df, _ = rolling_calls[0]
    assert set(passed_team_df["gameid"].unique()) == {"new_game"}
    assert result["gameid"].tolist() == ["new_game"]

