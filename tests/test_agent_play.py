from __future__ import annotations

import json

from jackbot.training import agent_play
from jackbot.training.advisor import AdvisorRecommendation, load_session


def run(session, *arguments: str) -> int:
    return agent_play.main(["--session", str(session), *arguments])


def test_noninteractive_game_survives_separate_commands(tmp_path, capsys) -> None:
    session_path = tmp_path / "table.json"

    assert run(session_path, "new", "--seat", "P1", "--hand", "AS KD 7H 4C") == 0
    capsys.readouterr()
    assert run(session_path, "moves") == 0
    assert "AS: spawn P1m1" in capsys.readouterr().out

    assert run(session_path, "observe", "--move", "1") == 0
    first = load_session(session_path)
    assert first.current_player == 1
    assert first.hand_sizes == [3, 4, 4, 4]
    assert len(first.my_hand) == 3

    assert run(session_path, "moves", "--card", "KH") == 0
    assert "KH: spawn P2m1" in capsys.readouterr().out
    assert run(session_path, "observe", "--card", "KH", "--move", "2") == 0
    second = load_session(session_path)
    assert second.current_player == 2
    assert second.hand_sizes == [3, 3, 4, 4]

    assert run(session_path, "undo", "--steps", "2") == 0
    restored = load_session(session_path)
    assert restored.current_player == 0
    assert restored.hand_sizes == [4, 4, 4, 4]
    assert len(restored.my_hand) == 4


def test_json_status_is_machine_readable(tmp_path, capsys) -> None:
    session_path = tmp_path / "table.json"
    assert run(session_path, "new", "--seat", "P3", "--hand", "AS KD 7H 4C") == 0
    capsys.readouterr()

    assert agent_play.main(
        ["--session", str(session_path), "--json", "status"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["state"]["user_seat"] == "P3"
    assert payload["state"]["current_player"] == "P1"
    assert payload["state"]["board"]["P1"]["P1m1"] == "base"


def test_opponent_move_requires_a_revealed_card_without_mutating_state(
    tmp_path,
    capsys,
) -> None:
    session_path = tmp_path / "table.json"
    assert run(session_path, "new", "--seat", "P1", "--hand", "AS KD 7H 4C") == 0
    assert run(session_path, "observe", "--move", "1") == 0
    before = session_path.read_bytes()
    capsys.readouterr()

    assert run(session_path, "moves") == 2
    assert "supply the revealed card" in capsys.readouterr().err
    assert session_path.read_bytes() == before


def test_advise_can_atomically_apply_a_ranked_move(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    session_path = tmp_path / "table.json"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    assert run(session_path, "new", "--seat", "P1", "--hand", "AS KD 7H 4C") == 0
    capsys.readouterr()

    monkeypatch.setattr(agent_play, "load_model_policy", lambda _: (object(), {}))
    monkeypatch.setattr(agent_play, "training_device", lambda: object())
    monkeypatch.setattr(
        agent_play,
        "rank_advisor_actions",
        lambda *args, **kwargs: [
            AdvisorRecommendation(
                action_id=0,
                label="AS: spawn P1m1",
                prior=0.5,
                score=0.6,
                wins=1.2,
                rollouts=2,
            )
        ],
    )

    assert agent_play.main(
        [
            "--session",
            str(session_path),
            "--checkpoint",
            str(checkpoint),
            "advise",
            "--samples",
            "2",
            "--rollouts-per-sample",
            "1",
            "--max-steps",
            "1",
            "--apply-rank",
            "1",
        ]
    ) == 0
    session = load_session(session_path)
    assert session.current_player == 1
    assert session.hand_sizes[0] == 3
    assert "Applied rank 1 and saved" in capsys.readouterr().out
