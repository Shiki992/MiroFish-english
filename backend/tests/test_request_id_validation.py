import pytest

from app import create_app
from app.services.report_agent import ReportManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reports = tmp_path / "uploads" / "reports"
    reports.mkdir(parents=True)
    keep = tmp_path / "uploads" / "projects" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("important")
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(reports))
    return create_app().test_client(), keep


def test_path_like_report_id_cannot_delete_the_data_folder(setup):
    client, keep = setup

    response = client.delete("/api/report/..")

    assert response.status_code == 400
    assert keep.exists()


@pytest.mark.parametrize("value", ["../../outside", "..", "a/b", "a\\b", "a\x00b"])
def test_path_like_ids_in_json_are_rejected(setup, value):
    client, _ = setup

    response = client.post(
        "/api/simulation/interview",
        json={"simulation_id": value, "agent_id": 0, "prompt": "hi"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid identifier"


def test_path_like_ids_in_query_strings_are_rejected(setup):
    client, _ = setup

    assert client.get("/api/simulation/list?project_id=../x").status_code == 400


def test_normal_ids_and_other_fields_still_work(setup):
    client, _ = setup

    missing = client.get("/api/report/report_abc-123")
    assert missing.status_code != 400 or missing.get_json()["error"] != "Invalid identifier"

    # A prompt may legitimately contain path-like text.
    response = client.post(
        "/api/simulation/interview",
        json={"simulation_id": "sim_abc", "agent_id": 0, "prompt": "What about ../x?"},
    )
    assert response.get_json().get("error") != "Invalid identifier"
