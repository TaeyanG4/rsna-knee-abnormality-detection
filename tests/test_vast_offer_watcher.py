"""Offline safety checks for the one-lease Vast market watcher."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


WATCHER = Path(__file__).resolve().parents[1] / "tools" / "vast_offer_watcher.py"
spec = importlib.util.spec_from_file_location("vast_offer_watcher", WATCHER)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def offer(offer_id=101, country="JP", price=0.174, **changes):
    value = {
        "id": offer_id, "machine_id": offer_id + 1000,
        "geolocation": f"Test, {country}", "gpu_name": "RTX 3090", "num_gpus": 1,
        "rentable": True, "rented": False, "verification": "verified", "cpu_arch": "amd64",
        "is_bid": False, "gpu_frac": 1, "gpu_ram": 24576, "cpu_ram": 64000,
        "cpu_cores_effective": 28, "disk_space": 200, "duration": 500000,
        "direct_port_count": 2, "cuda_max_good": 13.2, "reliability": 0.995,
        "inet_down": 800, "inet_up": 400, "dph_total": price,
        "internet_down_cost_per_tb": 2.5, "internet_up_cost_per_tb": 4.0,
    }
    value.update(changes)
    return value


class FakeProvider:
    def __init__(self, key_path: Path, offers=None, balance=29.45):
        self.api_key_path = key_path
        self.current_offers = offers if offers is not None else [offer()]
        self.current_instances = []
        self.current_balance = balance
        self.create_calls = []
        self.stop_calls = []
        self.destroy_calls = []
        self.create_response = {"success": True, "new_contract": 555}
        self.stop_error = False

    def offers(self, query=None):
        return list(self.current_offers)

    def instances(self):
        return list(self.current_instances)

    def user(self):
        return {"credit": self.current_balance, "billing_creditonly": 1,
                "autobill_amount": None}

    def create(self, offer_id, config, label):
        self.create_calls.append((offer_id, label))
        return self.create_response

    def stop(self, instance_id):
        self.stop_calls.append(instance_id)
        if self.stop_error:
            raise mod.CliError("stop ambiguous")
        return {"success": True}

    def destroy(self, instance_id):
        self.destroy_calls.append(instance_id)
        return {"success": True}


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        key = root / "vastai" / "vast_api_key"
        key.parent.mkdir()
        key.write_text("x" * 32)
        script = root / "onstart.sh"
        script.write_bytes(b"#!/bin/bash\ntrue\n")
        self.config = {
            "enabled": True, "api_key_path": str(key), "onstart_path": str(script),
            "desktop_notifications": False,
            "api_key_sha256": hashlib.sha256(key.read_bytes()).hexdigest(),
            "onstart_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "image_digest": "pytorch/pytorch@sha256:" + "a" * 64,
            "disk_gb": 100, "countries": ["JP", "KR"], "excluded_machines": [27157, 78080],
            "min_gpu_ram_mb": 24000, "min_cpu_ram_mb": 32000, "min_cpu_cores": 12,
            "min_direct_ports": 1, "min_cuda": 12.4, "min_reliability": 0.99,
            "min_download_mbps": 400, "min_upload_mbps": 100,
            "min_duration_seconds": 345600, "max_total_hourly_usd": 0.175,
            "max_transfer_usd_per_tb": 5, "auto_destroy_unclaimed": True,
            "balance_field": "credit", "reserve_usd": 5, "initial_budget_usd": 1.5,
            "unattended_minutes": 90, "max_total_minutes": 360,
            "label_prefix": "offline-test",
        }
        self.provider = FakeProvider(key)
        self.watcher = mod.Watcher(self.config, self.provider, root / "state")
        self.now = dt.datetime(2026, 10, 8, 12, 0, tzinfo=dt.timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_filters_and_cheapest_country_tiebreak(self):
        bad = [
            offer(1, "TW"), offer(2, price=0.176), offer(3, rentable=False),
            offer(4, verification="unverified"), offer(5, cpu_ram=31000),
            offer(6, direct_port_count=0), offer(7, duration=100),
            offer(8, machine_id=27157), offer(9, is_bid=True),
        ]
        self.assertTrue(all(not mod.offer_is_eligible(o, self.config) for o in bad))
        self.assertEqual(mod.choose_offer([offer(11, "KR", 0.174), offer(12, "JP", 0.174)], self.config)["id"], 12)
        self.assertEqual(mod.choose_offer([offer(11, "KR", 0.173), offer(12, "JP", 0.174)], self.config)["id"], 11)

    def test_existing_instance_and_insufficient_balance_block_create(self):
        self.provider.current_instances = [{"id": 1}]
        self.assertEqual(self.watcher.once(self.now), "account-not-clear")
        self.provider.current_instances = []
        self.provider.current_balance = 6.49
        self.assertEqual(self.watcher.once(self.now), "account-not-clear")
        self.assertFalse(self.provider.create_calls)

    def test_supervisor_never_creates_from_watching_state(self):
        self.assertEqual(self.watcher.supervise_only(self.now), "supervisor-idle")
        self.assertEqual(self.provider.create_calls, [])

    def test_auto_billing_setting_blocks_create(self):
        original_user = self.provider.user
        self.provider.user = lambda: {**original_user(), "billing_creditonly": 0,
                                      "autobill_amount": 10}
        self.assertEqual(self.watcher.once(self.now), "account-not-clear")
        self.assertFalse(self.provider.create_calls)

    def test_offer_change_blocks_mutation(self):
        original = self.provider.offers
        calls = 0

        def changing(query=None):
            nonlocal calls
            calls += 1
            return original(query) if calls == 1 else [offer(rentable=False)]

        self.provider.offers = changing
        self.assertEqual(self.watcher.once(self.now), "offer-changed")
        self.assertFalse(self.provider.create_calls)

    def test_create_durable_intent_then_never_second_create(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.assertEqual(len(self.provider.create_calls), 1)
        self.assertEqual(self.watcher.state()["instance_id"], 555)
        self.provider.current_instances = [{"id": 555, "actual_status": "running", "intended_status": "running",
                                            "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "leased-watching")
        self.assertEqual(len(self.provider.create_calls), 1)

    def test_lost_receipt_reconciles_by_label_without_retry(self):
        self.provider.create_response = {"success": True}
        self.assertEqual(self.watcher.once(self.now), "create-uncertain")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "create-uncertain")
        self.assertEqual(len(self.provider.create_calls), 1)
        self.provider.current_instances = [{"id": 999, "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "leased-reconciled")
        self.assertEqual(len(self.provider.create_calls), 1)

    def test_lost_receipt_adopts_sole_same_machine_without_label(self):
        self.provider.create_response = {"success": True}
        self.assertEqual(self.watcher.once(self.now), "create-uncertain")
        self.provider.current_instances = [{"id": 999, "machine_id": 1101, "label": None,
                                            "actual_status": "running", "intended_status": "running"}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "leased-reconciled")
        self.assertEqual(self.watcher.state()["instance_id"], 999)
        self.assertEqual(self.watcher.state()["identity_source"], "sole_machine_without_label")
        self.assertEqual(len(self.provider.create_calls), 1)
        self.provider.current_instances[0]["actual_status"] = "stopped"
        self.provider.current_instances[0]["intended_status"] = "stopped"
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "destroy-requested")
        self.assertEqual(self.provider.destroy_calls, [999])

    def test_lost_receipt_does_not_adopt_different_machine(self):
        self.provider.create_response = {"success": True}
        self.assertEqual(self.watcher.once(self.now), "create-uncertain")
        self.provider.current_instances = [{"id": 999, "machine_id": 2222, "label": None}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "create-uncertain")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=16)), "create-uncertain")
        self.assertNotIn("instance_id", self.watcher.state())
        self.assertEqual(len(self.provider.create_calls), 1)

    def test_exact_machine_requirement_blocks_mismatched_label(self):
        self.provider.create_response = {"success": True}
        self.assertEqual(self.watcher.once(self.now), "create-uncertain")
        state = self.watcher.state()
        state["require_machine_identity"] = True
        self.watcher.save(state)
        self.provider.current_instances = [{"id": 999, "machine_id": 9999, "label": state["label"]}]
        self.assertEqual(self.watcher.supervise_only(self.now + dt.timedelta(minutes=1)), "create-uncertain")
        self.assertNotIn("instance_id", self.watcher.state())
        self.assertEqual(len(self.provider.create_calls), 1)
        self.provider.current_instances = [{"id": 999, "machine_id": 1101,
                                            "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "leased-reconciled")
        self.assertEqual(len(self.provider.create_calls), 1)

    def test_stop_requested_once_and_reconciled(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.provider.current_instances = [{"id": 555, "actual_status": "running", "intended_status": "running",
                                            "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=91)), "stop-requested")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=92)), "stop-uncertain")
        self.assertEqual(self.provider.stop_calls, [555])
        self.provider.current_instances[0]["actual_status"] = "stopped"
        self.provider.current_instances[0]["intended_status"] = "stopped"
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=93)), "destroy-requested")
        self.assertEqual(self.provider.stop_calls, [555])
        self.assertEqual(self.provider.destroy_calls, [555])

    def test_failed_stop_retries_only_after_backoff(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.provider.current_instances = [{"id": 555, "actual_status": "running",
                                            "intended_status": "running", "label": self.watcher.state()["label"]}]
        self.provider.stop_error = True
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=91)), "stop-uncertain")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=92)), "stop-uncertain")
        self.assertEqual(len(self.provider.stop_calls), 1)
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=93)), "stop-uncertain")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=95)), "stop-uncertain")
        self.assertEqual(self.provider.stop_calls, [555, 555, 555])
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=97)), "stop-uncertain")
        self.assertEqual(self.provider.stop_calls, [555, 555, 555])
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=110)), "stop-uncertain")
        self.assertEqual(self.provider.stop_calls, [555, 555, 555, 555])
        self.assertEqual(self.watcher.state()["stop_failed_alerted"], True)

    def test_destroy_continues_after_three_attempts(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.provider.current_instances = [{"id": 555, "actual_status": "stopped",
                                            "intended_status": "stopped", "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "destroy-requested")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=3)), "destroy-requested")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=5)), "destroy-requested")
        self.assertEqual(self.provider.destroy_calls, [555, 555, 555])
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=19)), "destroy-uncertain")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=20)), "destroy-requested")
        self.assertEqual(self.provider.destroy_calls, [555, 555, 555, 555])

    def test_missing_listing_recovers_and_exited_running_is_not_stopped(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "lease-missing")
        self.provider.current_instances = [{"id": 555, "actual_status": "exited",
                                            "intended_status": "running", "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "leased-watching")
        self.assertEqual(self.provider.destroy_calls, [])
        self.assertEqual(self.watcher.state()["status"], "leased")

    def test_existing_lease_supervised_after_future_create_disabled(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.config["enabled"] = False
        self.config["onstart_sha256"] = "wrong"
        self.provider.current_instances = [{"id": 555, "actual_status": "running",
                                            "intended_status": "running", "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=91)), "stop-requested")

    def test_unclaimed_destroy_reconciles_without_second_rental(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.provider.current_instances = [{"id": 555, "actual_status": "stopped",
                                            "intended_status": "stopped", "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "destroy-requested")
        self.provider.current_instances = []
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "destroy-unconfirmed")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=3)), "destroyed")
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=4)), "destroyed")
        self.assertEqual(len(self.provider.create_calls), 1)
        self.assertEqual(self.provider.destroy_calls, [555])

    def test_destroy_absence_must_be_consecutive(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        present = {"id": 555, "actual_status": "stopped", "intended_status": "stopped",
                   "label": self.watcher.state()["label"]}
        self.provider.current_instances = [present]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=1)), "destroy-requested")
        self.provider.current_instances = []
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "destroy-unconfirmed")
        self.provider.current_instances = [present]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2, seconds=30)), "destroy-uncertain")
        self.provider.current_instances = []
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=3)), "destroy-unconfirmed")
        self.assertEqual(self.watcher.state()["status"], "destroying")

    def test_retained_lease_is_not_destroyed(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        self.assertEqual(self.watcher.retain(555, self.now + dt.timedelta(minutes=1)), "retained")
        self.provider.current_instances = [{"id": 555, "actual_status": "stopped",
                                            "intended_status": "stopped", "label": self.watcher.state()["label"]}]
        self.assertEqual(self.watcher.once(self.now + dt.timedelta(minutes=2)), "stopped-retained")
        self.assertEqual(self.provider.destroy_calls, [])

    def test_selected_lease_stops_before_phase_credit_cap(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        state = self.watcher.state()
        state.update(baseline_credit_usd="29.45", phase_spend_cap_usd="1.50",
                     max_transfer_reserved_usd="0.15")
        self.watcher.save(state)
        self.provider.current_instances = [{"id": 555, "actual_status": "running",
                                            "intended_status": "running", "label": state["label"]}]
        self.provider.current_balance = 28.04
        self.assertEqual(self.watcher.supervise_only(self.now + dt.timedelta(minutes=1)), "stop-requested")
        self.assertEqual(self.provider.stop_calls, [555])

    def test_selected_lease_stops_on_projected_cost_before_billing(self):
        self.assertEqual(self.watcher.once(self.now), "leased")
        state = self.watcher.state()
        state.update(baseline_credit_usd="29.45", phase_spend_cap_usd="0.20",
                     max_transfer_reserved_usd="0.15")
        self.watcher.save(state)
        self.provider.current_instances = [{"id": 555, "actual_status": "running",
                                            "intended_status": "running", "label": state["label"]}]
        self.assertEqual(self.watcher.supervise_only(self.now + dt.timedelta(minutes=1)), "stop-requested")

    def test_public_observation_alerts_only_on_change(self):
        self.assertEqual(self.watcher.observe(self.now), "eligible")
        self.assertEqual(self.watcher.observe(self.now + dt.timedelta(minutes=1)), "eligible")
        events = (Path(self.temp.name) / "state" / "events.jsonl").read_text().splitlines()
        self.assertEqual(len(events), 1)
        self.assertEqual(json.loads(events[0])["kind"], "OFFER_AVAILABLE")

    def test_disabled_desktop_notifications_do_not_launch_msg(self):
        with patch.object(mod.subprocess, "run") as run:
            self.watcher.event("OFFER_AVAILABLE", self.now, offer_id=101)
        run.assert_not_called()

    def test_single_process_lock_acquires_and_releases(self):
        lock = Path(self.temp.name) / "watcher.lock"
        with mod.single_process(lock):
            self.assertTrue(lock.exists())
        with mod.single_process(lock):
            pass

    def test_official_cli_create_command_is_one_attempt(self):
        cli = mod.VastCLI(Path("vastai.exe"), 100, Path(self.config["api_key_path"]))
        completed = subprocess.CompletedProcess([], 0, '{"success":true,"new_contract":555}\n', "")
        with patch.object(mod.subprocess, "run", return_value=completed) as run:
            response = cli.create(101, self.config, "offline-test")
        self.assertEqual(response["new_contract"], 555)
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ["--raw", "--retry", "1"])
        self.assertIn("--cancel-unavail", command)
        self.assertIn("--direct", command)
        self.assertIn("--onstart", command)

    def test_cli_stop_text_and_isolated_key_environment(self):
        cli = mod.VastCLI(Path("vastai.exe"), 100, Path(self.config["api_key_path"]))
        completed = subprocess.CompletedProcess([], 0, "stopping instance 555.\n", "")
        with patch.dict("os.environ", {"VAST_API_KEY": "unrelated-secret"}), patch.object(mod.subprocess, "run", return_value=completed) as run:
            self.assertEqual(cli.stop(555), {"success": True})
            kwargs = run.call_args.kwargs
            self.assertNotIn("VAST_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["env"]["XDG_CONFIG_HOME"], str(Path(self.config["api_key_path"]).parent.parent))


if __name__ == "__main__":
    unittest.main()
