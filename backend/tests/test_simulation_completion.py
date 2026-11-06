"""A run must finish after its last round while the process stays up for interviews."""
import threading

import pytest

from app.services import simulation_runner as runner_module
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunner,
    SimulationRunState,
)

SIMULATION_ID = "sim_completion"


class FakeProcess:
    """Stays alive until exit() is called."""

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def exit(self, code=0):
        self.returncode = code


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", lambda *a, **k: None)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    (tmp_path / SIMULATION_ID).mkdir()

    state = SimulationRunState(
        simulation_id=SIMULATION_ID,
        runner_status=RunnerStatus.RUNNING,
        twitter_running=True,
        reddit_running=True,
    )
    SimulationRunner._run_states[SIMULATION_ID] = state
    process = FakeProcess()
    SimulationRunner._processes[SIMULATION_ID] = process
    SimulationRunner._graph_memory_enabled.pop(SIMULATION_ID, None)
    SimulationRunner._manual_stop_requests.discard(SIMULATION_ID)
    monkeypatch.setattr(
        SimulationRunner, "_check_all_platforms_completed", lambda state: finished.is_set()
    )
    finished = threading.Event()
    yield state, process, finished
    SimulationRunner._run_states.pop(SIMULATION_ID, None)
    SimulationRunner._processes.pop(SIMULATION_ID, None)
    SimulationRunner._graph_memory_enabled.pop(SIMULATION_ID, None)
    SimulationRunner._manual_stop_requests.discard(SIMULATION_ID)


def status():
    return SimulationRunner.get_run_state(SIMULATION_ID).runner_status


def run_monitor():
    SimulationRunner._monitor_simulation(SIMULATION_ID)


def start_monitor():
    thread = threading.Thread(target=run_monitor, daemon=True)
    thread.start()
    return thread


def wait_for(condition, timeout=5):
    event = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if condition():
            return True
        event.wait(0.01)
    return False


def test_finished_rounds_publish_completed_while_process_stays_alive(run):
    state, process, finished = run
    finished.set()

    monitor = start_monitor()

    assert wait_for(lambda: status() == RunnerStatus.COMPLETED)
    assert process.poll() is None
    assert SimulationRunner.has_live_process(SIMULATION_ID)
    assert state.twitter_running is False and state.reddit_running is False
    assert state.completed_at

    process.exit(0)
    monitor.join(5)
    assert not monitor.is_alive()
    assert status() == RunnerStatus.COMPLETED


def test_run_is_not_complete_before_its_rounds_finish(run):
    _, process, finished = run

    monitor = start_monitor()

    assert not wait_for(lambda: status() == RunnerStatus.COMPLETED, timeout=0.1)
    finished.set()
    assert wait_for(lambda: status() == RunnerStatus.COMPLETED)
    process.exit(0)
    monitor.join(5)


def test_completion_waits_for_graph_writes(run, monkeypatch):
    _, process, finished = run
    SimulationRunner._graph_memory_enabled[SIMULATION_ID] = True
    release = threading.Event()
    seen = []

    def stop_updater(simulation_id):
        seen.append(status())
        release.wait(5)

    monkeypatch.setattr(runner_module.GraphMemoryManager, "stop_updater", stop_updater)
    finished.set()

    monitor = start_monitor()

    assert wait_for(lambda: seen == [RunnerStatus.STOPPING])
    assert status() == RunnerStatus.STOPPING
    release.set()
    assert wait_for(lambda: status() == RunnerStatus.COMPLETED)
    assert process.poll() is None
    process.exit(0)
    monitor.join(5)


def test_failed_graph_writes_fail_the_run_and_are_not_overwritten_on_exit(run, monkeypatch):
    _, process, finished = run
    SimulationRunner._graph_memory_enabled[SIMULATION_ID] = True

    def stop_updater(simulation_id):
        raise RuntimeError("episode failed")

    monkeypatch.setattr(runner_module.GraphMemoryManager, "stop_updater", stop_updater)
    finished.set()

    monitor = start_monitor()

    assert wait_for(lambda: status() == RunnerStatus.FAILED)
    process.exit(0)
    monitor.join(5)
    assert status() == RunnerStatus.FAILED
    assert "episode failed" in SimulationRunner.get_run_state(SIMULATION_ID).error


def test_process_dying_after_completion_does_not_fail_a_finished_run(run):
    _, process, finished = run
    finished.set()

    monitor = start_monitor()

    assert wait_for(lambda: status() == RunnerStatus.COMPLETED)
    process.exit(-9)
    monitor.join(5)
    assert status() == RunnerStatus.COMPLETED


def test_crash_before_the_last_round_is_still_a_failure(run):
    _, process, _ = run
    process.exit(-9)

    run_monitor()

    assert status() == RunnerStatus.FAILED
    assert "out of memory" in SimulationRunner.get_run_state(SIMULATION_ID).error


def test_stop_ends_the_waiting_process_of_a_completed_run(run, monkeypatch):
    _, process, finished = run
    finished.set()

    def terminate(target, simulation_id, timeout=10):
        target.terminated = True
        target.exit(-15)

    monkeypatch.setattr(SimulationRunner, "_terminate_process", terminate)
    monitor = start_monitor()
    SimulationRunner._monitor_threads[SIMULATION_ID] = monitor
    assert wait_for(lambda: status() == RunnerStatus.COMPLETED)

    result = SimulationRunner.stop_simulation(SIMULATION_ID)

    assert process.terminated is True
    assert result.runner_status == RunnerStatus.STOPPED
    monitor.join(5)
    assert not monitor.is_alive()


def test_stop_still_rejects_a_completed_run_with_no_process(run):
    state, _, _ = run
    state.runner_status = RunnerStatus.COMPLETED
    SimulationRunner._processes.pop(SIMULATION_ID)

    with pytest.raises(ValueError, match="not running"):
        SimulationRunner.stop_simulation(SIMULATION_ID)


def test_cannot_start_a_second_process_while_the_first_is_alive(run, tmp_path, monkeypatch):
    state, _, _ = run
    state.runner_status = RunnerStatus.COMPLETED
    monkeypatch.setattr(SimulationRunner, "get_run_state", lambda _id: state)
    config = tmp_path / SIMULATION_ID / "simulation_config.json"
    config.write_text('{"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}', encoding="utf-8")

    with pytest.raises(ValueError, match="already running or finalizing"):
        SimulationRunner.start_simulation(SIMULATION_ID)
