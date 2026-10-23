#!/usr/bin/env python3
"""Watch a bounded Vast market and claim one offer at most once.

The watcher uses the official vastai CLI. Its configuration contains no secret;
the CLI reads its scoped key from the user's normal config directory. A task
scheduler can call ``--once`` every minute. No GPU experiment is launched here.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from typing import Any


UTC = dt.timezone.utc
QUERY = "gpu_name=RTX_3090 num_gpus=1 geolocation in [JP,KR]"


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("missing or invalid numeric value") from None


def country(offer: dict[str, Any]) -> str:
    return str(offer.get("geolocation", "")).rsplit(",", 1)[-1].strip()


def offer_is_eligible(offer: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Fail closed on absent values; prices include the requested disk size."""
    try:
        if country(offer) not in policy["countries"]:
            return False
        if str(offer["gpu_name"]).replace("_", " ") != "RTX 3090":
            return False
        if int(offer["num_gpus"]) != 1:
            return False
        if offer["rentable"] is not True or offer["rented"] is not False:
            return False
        if offer["verification"] != "verified":
            return False
        if offer["cpu_arch"] != "amd64" or offer["is_bid"] is not False:
            return False
        if decimal(offer["gpu_frac"]) != 1:
            return False
        if int(offer["machine_id"]) in policy["excluded_machines"]:
            return False
        floors = {
            "gpu_ram": "min_gpu_ram_mb",
            "cpu_ram": "min_cpu_ram_mb",
            "cpu_cores_effective": "min_cpu_cores",
            "disk_space": "disk_gb",
            "duration": "min_duration_seconds",
            "direct_port_count": "min_direct_ports",
            "cuda_max_good": "min_cuda",
            "reliability": "min_reliability",
            "inet_down": "min_download_mbps",
            "inet_up": "min_upload_mbps",
        }
        if any(decimal(offer[field]) < decimal(policy[key]) for field, key in floors.items()):
            return False
        if decimal(offer["dph_total"]) > decimal(policy["max_total_hourly_usd"]):
            return False
        if decimal(offer["internet_down_cost_per_tb"]) > decimal(policy["max_transfer_usd_per_tb"]):
            return False
        if decimal(offer["internet_up_cost_per_tb"]) > decimal(policy["max_transfer_usd_per_tb"]):
            return False
        return decimal(offer["dph_total"]) > 0
    except (KeyError, TypeError, ValueError):
        return False


def choose_offer(offers: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
    eligible = [offer for offer in offers if offer_is_eligible(offer, policy)]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda offer: (
            decimal(offer["dph_total"]),
            0 if country(offer) == "JP" else 1,
            -decimal(offer["cpu_cores_effective"]),
            -decimal(offer["cpu_ram"]),
            -decimal(offer["reliability"]),
            int(offer["id"]),
        ),
    )


def extract_list(value: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get(field)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"unexpected {field} response shape")
    return value


class CliError(RuntimeError):
    pass


class VastCLI:
    def __init__(self, executable: Path, disk_gb: int, api_key_path: Path | None = None):
        self.executable = executable
        self.disk_gb = disk_gb
        self.api_key_path = api_key_path

    def call(self, *args: str, timeout: int = 45) -> Any:
        command = [str(self.executable), "--raw", "--retry", "1", *args]
        environment = os.environ.copy()
        # A shell's unrelated key must never outrank the dedicated watcher key.
        environment.pop("VAST_API_KEY", None)
        if self.api_key_path is not None:
            if self.api_key_path.name != "vast_api_key" or self.api_key_path.parent.name != "vastai":
                raise CliError("API key path must end in vastai/vast_api_key")
            environment["XDG_CONFIG_HOME"] = str(self.api_key_path.parent.parent)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout, shell=False, check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            # CLI output can include account data, so the exception is not logged.
            raise CliError(f"{args[0]} {args[1]} did not complete") from None
        if result.returncode != 0:
            raise CliError(f"{args[0]} {args[1]} exit={result.returncode}")
        if args[:2] == ("stop", "instance"):
            if result.stdout.strip() != f"stopping instance {args[2]}.":
                raise CliError("stop instance returned unconfirmed response")
            return {"success": True}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise CliError(f"{args[0]} {args[1]} returned non-JSON output") from None

    def offers(self, query: str = QUERY) -> list[dict[str, Any]]:
        response = self.call(
            "search", "offers", query, "--no-default", "--type", "on-demand",
            "--storage", str(self.disk_gb), "--order", "dph_total", "--limit", "300",
        )
        return extract_list(response, "offers")

    def instances(self) -> list[dict[str, Any]]:
        return extract_list(self.call("show", "instances"), "instances")

    def user(self) -> dict[str, Any]:
        value = self.call("show", "user")
        if not isinstance(value, dict):
            raise CliError("show user returned unexpected shape")
        return value

    def create(self, offer_id: int, config: dict[str, Any], label: str) -> dict[str, Any]:
        value = self.call(
            "create", "instance", str(offer_id), "--image", config["image_digest"],
            "--disk", str(config["disk_gb"]), "--label", label,
            "--onstart", str(config["onstart_path"]), "--ssh", "--direct",
            "--cancel-unavail", timeout=120,
        )
        if not isinstance(value, dict):
            raise CliError("create instance returned unexpected shape")
        return value

    def stop(self, instance_id: int) -> Any:
        return self.call("stop", "instance", str(instance_id), timeout=60)

    def destroy(self, instance_id: int) -> Any:
        response = self.call("destroy", "instance", str(instance_id), "--yes", timeout=60)
        if not isinstance(response, dict) or response.get("success") is not True:
            raise CliError("destroy instance returned unconfirmed response")
        return response


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return dict(default)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@contextlib.contextmanager
def single_process(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) != b"1":
            stream.seek(0)
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise CliError("another watcher process holds the lock") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Watcher:
    def __init__(self, config: dict[str, Any], provider: VastCLI, state_dir: Path):
        self.config = config
        self.provider = provider
        self.state_dir = state_dir
        self.state_path = state_dir / "state.json"
        self.alert_path = state_dir / "alert.json"
        self.events_path = state_dir / "events.jsonl"

    def event(self, kind: str, now: dt.datetime, **fields: Any) -> None:
        record = {"kind": kind, "at_utc": iso(now), **fields}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if kind in {"OFFER_AVAILABLE", "LEASED", "CREATE_UNCERTAIN", "STOP_REQUESTED", "STOP_UNCERTAIN", "STOP_FAILED", "STOPPED", "DESTROY_REQUESTED", "DESTROY_UNCERTAIN", "DESTROYED", "BLOCKED"}:
            atomic_json(self.alert_path, record)
            if os.name == "nt" and self.config.get("desktop_notifications") is True:
                try:
                    subprocess.run(["msg.exe", getpass.getuser(), "/TIME:30",
                                    f"RSNA Vast watcher: {kind}; instance {fields.get('instance_id', 'pending')}. See alert.json."],
                                   capture_output=True, timeout=5, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def state(self) -> dict[str, Any]:
        return read_json(self.state_path, {"status": "watching", "failed_offers": []})

    def save(self, state: dict[str, Any]) -> None:
        atomic_json(self.state_path, state)

    def observe(self, now: dt.datetime | None = None) -> str:
        """Public inventory transition monitor; no account key or mutation."""
        now = now or utc_now()
        offers = self.provider.offers()
        if len(offers) >= 300:
            raise ValueError("offer search hit result limit")
        selected = choose_offer(offers, self.config)
        snapshot = ({"offer_id": int(selected["id"]), "country": country(selected),
                     "quoted_hourly_usd": str(selected["dph_total"])} if selected else
                    {"offer_id": None})
        path = self.state_dir / "observation.json"
        previous = read_json(path, {})
        if previous != snapshot:
            atomic_json(path, snapshot)
            if selected:
                self.event("OFFER_AVAILABLE", now, **snapshot)
        return "eligible" if selected else "no-eligible-offer"

    def balance(self) -> Decimal:
        field = self.config.get("balance_field")
        if not isinstance(field, str) or not field:
            raise ValueError("balance_field must be verified before arming")
        user = self.provider.user()
        return decimal(user[field])

    def validate_static(self) -> None:
        if not self.config.get("enabled"):
            raise ValueError("watcher is disabled in config")
        self.validate_supervision()

        script = Path(self.config["onstart_path"])
        if hashlib.sha256(script.read_bytes()).hexdigest() != self.config["onstart_sha256"]:
            raise ValueError("on-start script hash changed")
        if "@sha256:" not in self.config["image_digest"]:
            raise ValueError("image must be digest-pinned")
        if decimal(self.config["max_total_hourly_usd"]) > Decimal("0.175"):
            raise ValueError("hourly cap exceeds the reviewed ceiling")
        if int(self.config["disk_gb"]) != 100:
            raise ValueError("disk size differs from reviewed 100 GB")
        if decimal(self.config["reserve_usd"]) < 5 or decimal(self.config["initial_budget_usd"]) > Decimal("1.50"):
            raise ValueError("budget exceeds reviewed limits")
        if not 10 <= int(self.config["unattended_minutes"]) <= 90:
            raise ValueError("unattended deadline must be within 10–90 minutes")
        if not int(self.config["unattended_minutes"]) <= int(self.config["max_total_minutes"]) <= 360:
            raise ValueError("lease extension limit exceeds six hours")
        if decimal(self.config["max_total_hourly_usd"]) * decimal(self.config["max_total_minutes"]) / 60 > Decimal("1.10"):
            raise ValueError("maximum compute time leaves too little initial-budget margin")
        if decimal(self.config["max_transfer_usd_per_tb"]) > 5:
            raise ValueError("transfer price exceeds reviewed ceiling")
        if self.config.get("auto_destroy_unclaimed") is not True:
            raise ValueError("unclaimed storage cleanup is not armed")

    def validate_supervision(self) -> None:
        """Keep an existing lease supervised even if future-create policy is disabled."""
        key_path = Path(self.config["api_key_path"])
        if self.provider.api_key_path != key_path:
            raise ValueError("CLI key binding differs from configured key")
        if not key_path.is_file() or key_path.stat().st_size < 20:
            raise ValueError("scoped API key file is absent")
        if hashlib.sha256(key_path.read_bytes()).hexdigest() != self.config.get("api_key_sha256"):
            raise ValueError("watcher API key changed")
        if not isinstance(self.config.get("balance_field"), str) or not self.config["balance_field"]:
            raise ValueError("live account balance field is unverified")

    def _account_clear(self) -> bool:
        instances = self.provider.instances()
        if instances:
            return False
        user = self.provider.user()
        if user.get("billing_creditonly") != 1 or user.get("autobill_amount") not in (None, 0):
            return False
        needed = decimal(self.config["reserve_usd"]) + decimal(self.config["initial_budget_usd"])
        return decimal(user[self.config["balance_field"]]) >= needed

    def once(self, now: dt.datetime | None = None) -> str:
        now = now or utc_now()
        state = self.state()
        status = state.get("status")
        if status == "watching":
            self.validate_static()
        else:
            self.validate_supervision()
        if status == "leased":
            return self._watch_lease(state, now)
        if status in {"creating", "create_uncertain", "stop_uncertain", "stopping", "destroying", "destroy_uncertain"}:
            return self._reconcile(state, now)
        if status == "stopped":
            return self._watch_stopped(state, now)
        if status in {"destroyed", "blocked"}:
            return str(status)
        if status != "watching":
            raise ValueError("unknown watcher state")
        if not self._account_clear():
            return "account-not-clear"
        offers = self.provider.offers()
        if len(offers) >= 300:
            raise ValueError("offer search hit result limit")
        failed = set(state.get("failed_offers", []))
        candidate = choose_offer([o for o in offers if o.get("id") not in failed], self.config)
        if candidate is None:
            return "no-eligible-offer"
        # Re-quote the exact machine just before one billable request.
        machine = int(candidate["machine_id"])
        fresh = self.provider.offers(f"machine_id={machine}")
        exact = [o for o in fresh if int(o.get("id", -1)) == int(candidate["id"])]
        if len(exact) != 1 or not offer_is_eligible(exact[0], self.config):
            return "offer-changed"
        if not self._account_clear():
            return "account-changed"
        label = f"{self.config['label_prefix']}-{now.strftime('%Y%m%dT%H%M%SZ')}-{candidate['id']}"
        state = {
            "status": "creating", "offer_id": int(candidate["id"]),
            "machine_id": machine, "country": country(candidate), "label": label,
            "quoted_hourly_usd": str(candidate["dph_total"]),
            "create_started_utc": iso(now), "failed_offers": sorted(failed),
            "retained": False,
        }
        self.save(state)  # Durable intent exists before the paid mutation.
        self.event("CREATE_ATTEMPT", now, offer_id=state["offer_id"], country=state["country"])
        try:
            response = self.provider.create(state["offer_id"], self.config, label)
            instance_id = int(response["new_contract"])
            if response.get("success") is not True or instance_id <= 0:
                raise ValueError("create response lacked a confirmed new contract")
        except (CliError, KeyError, TypeError, ValueError):
            state["status"] = "create_uncertain"
            state["last_create_uncertain_alert_utc"] = iso(now)
            self.save(state)
            self.event("CREATE_UNCERTAIN", now, offer_id=state["offer_id"])
            return "create-uncertain"
        state["status"] = "leased"
        state["instance_id"] = instance_id
        state["identity_source"] = "create_receipt"
        state["created_utc"] = iso(now)
        state["deadline_utc"] = iso(now + dt.timedelta(minutes=int(self.config["unattended_minutes"])))
        self.save(state)
        self.event("LEASED", now, offer_id=state["offer_id"], instance_id=instance_id,
                   country=state["country"], quoted_hourly_usd=state["quoted_hourly_usd"],
                   deadline_utc=state["deadline_utc"])
        return "leased"

    def supervise_only(self, now: dt.datetime | None = None) -> str:
        """Supervise a preselected lease; never enter the market/create branch."""
        if self.state().get("status") == "watching":
            self.validate_supervision()
            return "supervisor-idle"
        return self.once(now)

    def _reconcile(self, state: dict[str, Any], now: dt.datetime) -> str:
        instances = self.provider.instances()
        matching = [i for i in instances if i.get("label") == state.get("label")
                    and self._machine_matches(i, state)]
        if state["status"] in {"creating", "create_uncertain"}:
            source = "label"
            if len(matching) == 1:
                found = matching[0]
            elif (len(instances) == 1
                  and instances[0].get("label") in (None, "")
                  and self._machine_matches(instances[0], state)
                  and str(instances[0].get("machine_id")) == str(state["machine_id"])):
                # The account was empty immediately before this one create.
                # Adopt a sole same-machine contract even if its label was lost.
                found = instances[0]
                source = "sole_machine_without_label"
            else:
                found = None
            if found is not None:
                state["instance_id"] = int(found["id"])
                state["identity_source"] = source
                state["status"] = "leased"
                state["created_utc"] = state["create_started_utc"]
                started = dt.datetime.fromisoformat(state["created_utc"])
                state["deadline_utc"] = iso(started + dt.timedelta(minutes=int(self.config["unattended_minutes"])))
                self.save(state)
                self.event("LEASED", now, offer_id=state["offer_id"],
                           instance_id=state["instance_id"], country=state["country"],
                           quoted_hourly_usd=state["quoted_hourly_usd"], deadline_utc=state["deadline_utc"],
                           identity_source=source)
                return "leased-reconciled"
            last_alert = state.get("last_create_uncertain_alert_utc")
            if last_alert is None or now - dt.datetime.fromisoformat(last_alert) >= dt.timedelta(minutes=15):
                state["last_create_uncertain_alert_utc"] = iso(now)
                self.save(state)
                self.event("CREATE_UNCERTAIN", now, offer_id=state["offer_id"],
                           account_instance_count=len(instances))
            return "create-uncertain"  # Never reissue create on absent/ambiguous receipt.
        if state["status"] in {"destroying", "destroy_uncertain"}:
            own = [i for i in instances if int(i.get("id", -1)) == int(state["instance_id"])]
            if not own:
                state["destroy_absent_count"] = int(state.get("destroy_absent_count", 0)) + 1
                self.save(state)
                if state["destroy_absent_count"] >= 2:
                    state["status"] = "destroyed"
                    self.save(state)
                    self.event("DESTROYED", now, instance_id=state["instance_id"])
                    return "destroyed"
                return "destroy-unconfirmed"
            if not self._identity_matches(own[0], state):
                self.event("BLOCKED", now, reason="owned instance identity changed", instance_id=state["instance_id"])
                return "destroy-label-changed"
            if state.pop("destroy_absent_count", 0):
                self.save(state)
            retry_at = dt.datetime.fromisoformat(state["next_destroy_retry_utc"])
            if now >= retry_at:
                if int(state.get("destroy_attempts", 0)) >= 3 and not state.get("destroy_failed_alerted"):
                    state["destroy_failed_alerted"] = True
                    self.save(state)
                    self.event("BLOCKED", now, reason="destroy still unconfirmed; continuing 15-minute retries",
                               instance_id=state["instance_id"])
                return self._request_destroy(state, now)
            return "destroy-uncertain"
        own = [i for i in instances if int(i.get("id", -1)) == int(state["instance_id"])]
        if len(own) == 1 and not self._identity_matches(own[0], state):
            self.event("BLOCKED", now, reason="owned instance identity changed", instance_id=state["instance_id"])
            return "stop-identity-changed"
        if len(own) == 1 and self._is_stopped(own[0]):
            state["status"] = "stopped"
            self.save(state)
            self.event("STOPPED", now, instance_id=state["instance_id"])
            return self._watch_stopped(state, now)
        retry_at = dt.datetime.fromisoformat(state["next_stop_retry_utc"])
        if now >= retry_at:
            if int(state.get("stop_attempts", 0)) >= 3 and not state.get("stop_failed_alerted"):
                state["stop_failed_alerted"] = True
                self.save(state)
                self.event("STOP_FAILED", now, instance_id=state["instance_id"], attempts=3,
                           reason="continuing 15-minute retries")
            return self._request_stop(state, now)
        return "stop-uncertain"

    @staticmethod
    def _machine_matches(instance: dict[str, Any], state: dict[str, Any]) -> bool:
        if not state.get("require_machine_identity"):
            return True
        return str(instance.get("machine_id")) == str(state.get("machine_id"))

    @staticmethod
    def _identity_matches(instance: dict[str, Any], state: dict[str, Any]) -> bool:
        if int(instance.get("id", -1)) != int(state["instance_id"]):
            return False
        if not Watcher._machine_matches(instance, state):
            return False
        if instance.get("label") == state.get("label"):
            return True
        return (instance.get("label") in (None, "")
                and str(instance.get("machine_id")) == str(state["machine_id"]))

    @staticmethod
    def _is_stopped(instance: dict[str, Any]) -> bool:
        return (str(instance.get("actual_status", "")).lower() in {"stopped", "exited"}
                and str(instance.get("intended_status", "")).lower() == "stopped")

    def _watch_stopped(self, state: dict[str, Any], now: dt.datetime) -> str:
        if state.get("retained"):
            return "stopped-retained"
        own = [i for i in self.provider.instances() if int(i.get("id", -1)) == int(state["instance_id"])]
        if len(own) != 1 or not self._identity_matches(own[0], state) or not self._is_stopped(own[0]):
            self.event("BLOCKED", now, reason="unclaimed cleanup identity/status unconfirmed",
                       instance_id=state["instance_id"])
            return "destroy-blocked"
        return self._request_destroy(state, now)

    def _request_destroy(self, state: dict[str, Any], now: dt.datetime) -> str:
        # An unretained lease never receives research inputs or unique outputs.
        state["status"] = "destroying"
        state["destroy_attempts"] = int(state.get("destroy_attempts", 0)) + 1
        state["next_destroy_retry_utc"] = iso(now + dt.timedelta(
            minutes=2 if state["destroy_attempts"] < 3 else 15))
        self.save(state)
        try:
            self.provider.destroy(int(state["instance_id"]))
            self.event("DESTROY_REQUESTED", now, instance_id=state["instance_id"],
                       attempt=state["destroy_attempts"])
            return "destroy-requested"
        except CliError:
            state["status"] = "destroy_uncertain"
            self.save(state)
            self.event("DESTROY_UNCERTAIN", now, instance_id=state["instance_id"],
                       attempt=state["destroy_attempts"])
            return "destroy-uncertain"

    def _request_stop(self, state: dict[str, Any], now: dt.datetime) -> str:
        state["status"] = "stopping"
        state["stop_attempts"] = int(state.get("stop_attempts", 0)) + 1
        state["next_stop_retry_utc"] = iso(now + dt.timedelta(
            minutes=2 if state["stop_attempts"] < 3 else 15))
        self.save(state)  # Persist each idempotent stop attempt before the call.
        try:
            self.provider.stop(int(state["instance_id"]))
            self.event("STOP_REQUESTED", now, instance_id=state["instance_id"],
                       attempt=state["stop_attempts"])
            return "stop-requested"
        except CliError:
            state["status"] = "stop_uncertain"
            self.save(state)
            self.event("STOP_UNCERTAIN", now, instance_id=state["instance_id"],
                       attempt=state["stop_attempts"])
            return "stop-uncertain"

    def _watch_lease(self, state: dict[str, Any], now: dt.datetime) -> str:
        instances = self.provider.instances()
        own = [i for i in instances if int(i.get("id", -1)) == int(state["instance_id"])]
        if len(own) != 1:
            state["missing_count"] = int(state.get("missing_count", 0)) + 1
            self.save(state)
            if state["missing_count"] == 3:
                self.event("BLOCKED", now, reason="leased instance absent from three account listings",
                           instance_id=state["instance_id"])
            if now >= dt.datetime.fromisoformat(state["deadline_utc"]):
                return self._request_stop(state, now)
            return "lease-missing"
        if not self._identity_matches(own[0], state):
            self.event("BLOCKED", now, reason="owned instance identity changed", instance_id=state["instance_id"])
            return "lease-identity-changed"
        if state.pop("missing_count", 0):
            self.save(state)
        if self._is_stopped(own[0]):
            state["status"] = "stopped"
            self.save(state)
            self.event("STOPPED", now, instance_id=state["instance_id"])
            return self._watch_stopped(state, now)
        balance = self.balance()
        if state.get("baseline_credit_usd") is not None and state.get("phase_spend_cap_usd") is not None:
            cap = decimal(state["phase_spend_cap_usd"])
            billed = decimal(state["baseline_credit_usd"]) - balance
            elapsed_hours = decimal(max(0, (now - dt.datetime.fromisoformat(
                state["created_utc"])).total_seconds())) / 3600
            projected = (elapsed_hours * decimal(state["quoted_hourly_usd"])
                         + decimal(state.get("max_transfer_reserved_usd", 0)))
            if billed >= cap - Decimal("0.10") or projected >= cap - Decimal("0.10"):
                return self._request_stop(state, now)
        if (now < dt.datetime.fromisoformat(state["deadline_utc"])
                and balance > decimal(self.config["reserve_usd"]) + Decimal("0.10")):
            return "leased-watching"
        return self._request_stop(state, now)

    def retain(self, instance_id: int, now: dt.datetime | None = None) -> str:
        """Explicit transfer to the research supervisor before SSH or inputs."""
        now = now or utc_now()
        state = self.state()
        if state.get("status") != "leased" or int(state.get("instance_id", -1)) != instance_id:
            raise ValueError("only the exact confirmed lease can be retained")
        state["retained"] = True
        self.save(state)
        self.event("RETAINED", now, instance_id=instance_id)
        return "retained"

    def extend(self, minutes: int, now: dt.datetime | None = None) -> str:
        now = now or utc_now()
        state = self.state()
        if state.get("status") != "leased":
            raise ValueError("only a confirmed lease can be extended")
        if minutes <= 0:
            raise ValueError("extension must be positive")
        max_deadline = dt.datetime.fromisoformat(state["created_utc"]) + dt.timedelta(
            minutes=int(self.config["max_total_minutes"]))
        deadline = min(now + dt.timedelta(minutes=minutes), max_deadline)
        if deadline <= now:
            raise ValueError("reviewed maximum lease time has expired")
        state["deadline_utc"] = iso(deadline)
        self.save(state)
        self.event("EXTENDED", now, instance_id=state["instance_id"], deadline_utc=state["deadline_utc"])
        return state["deadline_utc"]


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    base = path.parent
    for key in ("cli_path", "onstart_path", "state_dir"):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (base / value).resolve())
    config["api_key_path"] = str(Path(config["api_key_path"]).expanduser())
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="one scheduled tick")
    group.add_argument("--supervise-only", action="store_true", help="supervise selected lease without market creation")
    group.add_argument("--dry-run", action="store_true", help="public inventory only; no account read or rental")
    group.add_argument("--observe", action="store_true", help="public inventory transition alert; no account read or rental")
    group.add_argument("--check-config", action="store_true", help="offline arming validation; no account read or rental")
    group.add_argument("--extend", type=int, metavar="MINUTES", help="extend confirmed lease within config maximum")
    group.add_argument("--retain", type=int, metavar="INSTANCE_ID", help="handoff exact lease before SSH or research")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        provider = VastCLI(Path(config["cli_path"]), int(config["disk_gb"]), Path(config["api_key_path"]))
        watcher = Watcher(config, provider, Path(config["state_dir"]))
        if args.check_config:
            watcher.validate_static()
            print("config-ready")
            return 0
        if args.dry_run:
            offers = provider.offers()
            choice = choose_offer(offers, config)
            print(json.dumps({"status": "eligible" if choice else "no-eligible-offer",
                              "listed": len(offers), "offer_id": choice.get("id") if choice else None,
                              "country": country(choice) if choice else None}))
            return 0
        if args.observe:
            with single_process(Path(config["state_dir"]) / "watcher.lock"):
                result = watcher.observe()
            print(result)
            return 0
        with single_process(Path(config["state_dir"]) / "watcher.lock"):
            if args.extend is not None:
                result = watcher.extend(args.extend)
            elif args.retain is not None:
                result = watcher.retain(args.retain)
            elif args.supervise_only:
                result = watcher.supervise_only()
            else:
                result = watcher.once()
        print(result)
        return 0
    except (CliError, KeyError, OSError, ValueError) as exc:
        # Never print provider stdout/stderr, credential values, or account objects.
        print(f"watcher error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
