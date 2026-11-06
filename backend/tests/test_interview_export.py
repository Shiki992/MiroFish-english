import csv
import io
import json
import sqlite3

import pytest
from flask import Flask

from app.api import simulation_bp
from app.api.simulation import (
    INTERVIEW_PROMPT_INTRO,
    LEGACY_INTERVIEW_PROMPT_PREFIX,
    _clean_interview_question,
    _csv_safe,
    optimize_interview_prompt,
)
from app.services.simulation_runner import SimulationRunner
from app.utils.locale import get_language_instruction, get_locale


def write_db(path, rows, with_trace=True):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE user (user_id INTEGER, user_name TEXT, name TEXT)")
    conn.execute("INSERT INTO user VALUES (1, 'ada_l', 'Ada')")
    if with_trace:
        conn.execute(
            "CREATE TABLE trace (user_id INTEGER, created_at, action TEXT, info TEXT)"
        )
        conn.executemany("INSERT INTO trace VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def interview(user_id, created_at, question, answer, interview_id=1):
    return (
        user_id,
        created_at,
        "interview",
        json.dumps(
            {
                "prompt": LEGACY_INTERVIEW_PROMPT_PREFIX + question,
                "response": answer,
                "interview_id": interview_id,
            }
        ),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))

    first = tmp_path / "sim_a"
    first.mkdir()
    write_db(
        first / "reddit_simulation.db",
        [
            interview(1, "2026-01-02", "Why?", "=SUM(1)"),
            (1, "2026-01-01", "create_post", "{}"),
        ],
    )
    write_db(
        first / "twitter_simulation.db",
        [
            # Twitter databases store simulation step numbers as timestamps.
            interview(9, 10, "How are you", "Very well", "10_0"),
            interview(9, 9, "Earlier?", "Yes", "9_0"),
        ],
    )

    second = tmp_path / "sim_b"
    second.mkdir()
    write_db(second / "reddit_simulation.db", [], with_trace=False)
    write_db(
        second / "twitter_simulation.db",
        [interview(1, "2026-01-03", "Later?", "Yes", 3)],
    )

    (tmp_path / "sim_empty").mkdir()

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    return app.test_client()


def parse_csv(response):
    return list(csv.DictReader(io.StringIO(response.data.decode("utf-8-sig"))))


def test_csv_export_for_one_simulation(client):
    response = client.get("/api/simulation/sim_a/interviews/export")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert 'filename="interviews_sim_a.csv"' in response.headers["Content-Disposition"]
    assert response.data.startswith(b"\xef\xbb\xbf")

    rows = parse_csv(response)
    assert [row["timestamp"] for row in rows] == ["2026-01-02", "9", "10"]

    assert rows[0]["platform"] == "reddit"
    assert rows[0]["agent_name"] == "Ada"
    assert rows[0]["question"] == "Why?"
    assert rows[0]["response"] == "'=SUM(1)"

    assert rows[2]["platform"] == "twitter"
    assert rows[2]["question"] == "How are you"
    assert rows[2]["agent_name"] == ""


def test_json_export_keeps_original_text(client):
    response = client.get("/api/simulation/sim_a/interviews/export?format=json")

    rows = json.loads(response.data)
    assert response.mimetype == "application/json"
    assert rows[0]["response"] == "=SUM(1)"
    assert rows[2]["response"] == "Very well"


def test_export_covers_all_simulations(client):
    response = client.get("/api/simulation/interviews/export")

    rows = parse_csv(response)
    assert 'filename="interviews_all_simulations.csv"' in response.headers["Content-Disposition"]
    assert [row["simulation_id"] for row in rows] == ["sim_a", "sim_a", "sim_a", "sim_b"]


def test_export_has_no_row_limit(client, tmp_path):
    many = tmp_path / "sim_many"
    many.mkdir()
    write_db(
        many / "reddit_simulation.db",
        [interview(1, f"2026-01-{i:04d}", "Q", "A", i) for i in range(250)],
    )

    response = client.get("/api/simulation/sim_many/interviews/export")

    assert len(parse_csv(response)) == 250


def test_simulation_without_interviews_exports_header_only(client):
    response = client.get("/api/simulation/sim_empty/interviews/export")

    assert response.status_code == 200
    assert parse_csv(response) == []


def test_missing_simulation_and_bad_format_are_rejected(client):
    assert client.get("/api/simulation/nope/interviews/export").status_code == 404
    assert client.get("/api/simulation/%2E%2E/interviews/export").status_code == 404
    assert client.get("/api/simulation/sim_a/interviews/export?format=xml").status_code == 400


def test_unreadable_database_fails_instead_of_exporting_partial_data(client, tmp_path):
    (tmp_path / "sim_a" / "reddit_simulation.db").write_bytes(b"not a database file" * 100)

    assert client.get("/api/simulation/sim_a/interviews/export").status_code == 500


def test_app_defaults_to_english():
    assert get_locale() == "en"
    assert get_language_instruction() == "Please respond in English."


def test_interview_prompt_is_english_and_export_recovers_the_question():
    prompt = optimize_interview_prompt("What changed?")

    assert prompt.startswith(INTERVIEW_PROMPT_INTRO)
    assert "Please respond in English." in prompt
    assert optimize_interview_prompt(prompt) == prompt
    assert _clean_interview_question(prompt) == "What changed?"
    assert _clean_interview_question(LEGACY_INTERVIEW_PROMPT_PREFIX + "Old question") == "Old question"
    assert _clean_interview_question("plain") == "plain"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ""),
        (3, 3),
        ("plain", "plain"),
        ("-x", "'-x"),
        ("@x", "'@x"),
        ({"a": "é"}, '{"a": "é"}'),
    ],
)
def test_csv_safe(value, expected):
    assert _csv_safe(value) == expected
