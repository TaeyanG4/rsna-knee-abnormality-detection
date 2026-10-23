"""EXP190 v1.4.0: private current-source registration; canonical binding stays strict."""
import copy
import json
import tempfile
import types
import unittest
import math
from pathlib import Path
from unittest.mock import patch

from tools import exp190_coatnet_oai_f0 as exp


class TestExp190(unittest.TestCase):
    def test_fixed_step_sampling_and_bn_replay(self):
        exp.fixed_step_checks()
        draws = exp.fixed_draw_counts(exp.bridge.N_TRAIN, 2000)
        self.assertEqual(draws["rsna"] + draws["oai"], 3455)
        self.assertEqual(draws["total_updates"], 3456)
        with self.assertRaises(ValueError):
            exp.fixed_draw_counts(4347, 2000)

    def test_epoch_assignment_reaches_wrapped_dataset(self):
        import torch
        class Dataset:
            def __init__(self):
                self.ids = [f"R{i}" for i in range(exp.bridge.N_TRAIN)] + ["OAI_1_R_V00"]
                self.rsna_ids = set(self.ids[:exp.bridge.N_TRAIN])
                self.oai_ids = set(self.ids[exp.bridge.N_TRAIN:])
                self.rsna = types.SimpleNamespace(augment=True, epoch=torch.zeros(1, dtype=torch.int64))
                self.oai = types.SimpleNamespace(augment=True, epoch=torch.zeros(1, dtype=torch.int64))

            def set_epoch(self, epoch):
                self.rsna.epoch.fill_(epoch)
                self.oai.epoch.fill_(epoch)

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, index):
                return self.ids[index]

        wrapped = exp.FixedStepMixedDataset(Dataset())
        wrapped.set_epoch(3)
        self.assertEqual(wrapped.inner.rsna.epoch.item(), 3)
        self.assertEqual(wrapped.inner.oai.epoch.item(), 3)
        self.assertEqual(wrapped._epoch(), 3)
        self.assertEqual(len(wrapped.selection(3)), exp.bridge.N_TRAIN)

    def test_recipe_changes_only_registered_oai_fields(self):
        control = json.loads(exp.bridge.manifest_path("CONTROL").read_text(encoding="utf-8"))
        control_recipe = control["recipe"]
        extension = {"oai_targets", "oai_mix", "loss", "class_weights", "bn_cohort"}
        allowed_shared = {"labels", "oai", "grad_checkpointing_default", "grad_checkpointing_allowed"}
        self.assertEqual(set(exp.old_oai.RECIPE) - set(control_recipe), extension)
        self.assertEqual({k for k in exp.old_oai.RECIPE.keys() & control_recipe.keys()
                          if exp.old_oai.RECIPE[k] != control_recipe[k]}, allowed_shared)
        self.assertFalse(control["grad_checkpointing"])
        recipe = exp.control_recipe_with_oai(control)
        for key, value in control_recipe.items():
            if key not in allowed_shared:
                self.assertEqual(recipe[key], value)
        self.assertEqual(recipe["oai_targets"], list(exp.OAI_TARGETS))
        bad = copy.deepcopy(control)
        bad["recipe"]["epochs"] = 15
        with self.assertRaises(ValueError):
            exp.control_recipe_with_oai(bad)
        bad = copy.deepcopy(control)
        bad["recipe"]["wd"] = 0.01
        with self.assertRaises(ValueError):
            exp.control_recipe_with_oai(bad)
        bad = copy.deepcopy(control)
        bad["recipe"]["unexpected_control_key"] = True
        self.assertEqual(exp.control_recipe_with_oai(bad), recipe)
        bad = copy.deepcopy(control)
        bad["grad_checkpointing"] = True
        with self.assertRaises(ValueError):
            exp.control_recipe_with_oai(bad)
        with patch.dict(exp.old_oai.RECIPE, {"unexpected_oai_key": True}):
            self.assertEqual(exp.control_recipe_with_oai(control), recipe)

    def test_explicit_api_leaves_module_globals_unchanged(self):
        import exp155_own0949_oai_v11 as fit
        self.assertEqual(exp.trainer_contract(fit), "exp155-mixed-v2")
        before = (fit.MixedStudyDataset, fit.mixed_dataset, exp.old_oai.load_registration)
        guard = exp.trainer_guard()
        self.assertIs(guard.load_registration, exp.load_registration)
        self.assertIs(guard.registered_run, exp.run_path)
        self.assertEqual(before, (fit.MixedStudyDataset, fit.mixed_dataset, exp.old_oai.load_registration))

    def test_completed_verification_receives_explicit_checks(self):
        from training.arms import coatnet_history
        self.assertFalse(hasattr(exp, "bridge_context"))
        before = (exp.bridge.load_registration, exp.bridge.check_summary, exp.bridge.EXP)
        p = {"arm": "OAI"}
        with patch.object(exp, "load_registration", return_value=p), \
             patch.object(exp, "require_validation"), \
             patch.object(exp, "HASH", return_value="protocol"), \
             patch.object(coatnet_history, "verify_record", return_value={}) as verify:
            self.assertFalse(exp.verified_arm()["historical_only"])
            verify.assert_called_once_with(p, exp.run_path(), "protocol",
                summary_checker=exp.check_summary, checkpoint_checker=exp.check_checkpoint)
        self.assertEqual(before, (exp.bridge.load_registration, exp.bridge.check_summary, exp.bridge.EXP))

    def test_real_cache_train_and_bn_dataloaders(self):
        """Exercise the same EXP155 factory and real cached volumes without CUDA."""
        import exp155_own0949_oai_v11 as fit
        from torch.utils.data import DataLoader

        from oai_registration_fixture import registration
        p = registration()
        args = types.SimpleNamespace(corpus=str(exp.bridge.CORPUS), oai_corpus=str(exp.CACHE))
        train = exp.dataset_factory(args, p, 42)
        bn = exp.dataset_factory(args, p, 42, augment=False)
        self.assertEqual((type(train.inner), type(bn.inner)), (fit.MixedStudyDataset, fit.MixedStudyDataset))
        self.assertTrue(train.augment)
        self.assertFalse(bn.augment)
        self.assertEqual((len(train), len(bn)), (3455, len(p["train_ids"])))
        self.assertEqual(train.draws, p["sampling"])
        self.assertEqual(math.ceil(len(train) / 16), 216)
        self.assertEqual(len(bn), len(p["train_ids"]))
        for epoch in (0, 1):
            train.set_epoch(epoch)
            selected = train.selection(epoch)
            self.assertEqual(sum(i in train.oai for i in selected), p["sampling"]["oai"])
            self.assertEqual(sum(i in train.rsna for i in selected), p["sampling"]["rsna"])
        train.set_epoch(0)
        bn.set_epoch(16)
        selected = train.selection(0)
        rsna_pos = next(i for i, j in enumerate(selected) if j in train.rsna)
        oai_pos = next(i for i, j in enumerate(selected) if j in train.oai)
        for dataset, indices in ((train, [rsna_pos, oai_pos]), (bn, [0, exp.bridge.N_TRAIN])):
            loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=2,
                                collate_fn=fit.collate, worker_init_fn=fit.worker_init)
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            for batch in batches:
                x, y, mask, ids = fit.unpack_batch(batch)
                self.assertEqual(x.shape[0], 1)
                self.assertEqual(y.shape, (1, 12))
                self.assertEqual(mask.shape, (1, 12))
                self.assertEqual(len(ids), 1)
            self.assertFalse(batches[0][3][0].startswith("OAI_"))
            self.assertTrue(batches[1][3][0].startswith("OAI_"))
        print(f"EXP190F real-cache smoke: draws={len(train)}, BN={len(bn)}, "
              "train_batches=2, BN_batches=2, workers=2, CPU PASS")

    def test_cache_rejects_unverified_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "corrected production v2 cache path"):
                exp.cache_binding(Path(tmp))

    def test_weak_four_screen_and_harm_guard(self):
        def pair(gain, dread):
            deltas = {target: gain for target in exp.bridge.TARGETS}
            return {"primary_mean_delta": gain,
                    "references": {"jev": {"per_target_delta": deltas},
                                   "consensus": {"per_target_delta": deltas},
                                   "dread": {"delta": dread}}}
        status, weak, _, _, checks = exp.screen_decision(pair(.003, 0.0))
        self.assertEqual((status, weak), ("REVIEW", .003))
        self.assertTrue(all(checks.values()))
        self.assertEqual(exp.screen_decision(pair(.003, -.004))[0], "HOLD")
        self.assertEqual(exp.screen_decision(pair(0.0, 0.0))[0], "HOLD")

    def test_oai_diagnostics_preserve_shared_scores_without_automatic_decision(self):
        from training.arms import coatnet_screen
        deltas = {target: (i + 1) / 10000 for i, target in enumerate(exp.bridge.TARGETS)}
        primary = dict(primary_mean_delta=.003, references={
            "jev": {"per_target_delta": deltas}, "consensus": {"per_target_delta": deltas},
            "dread": {"delta": 0.0}})
        common = dict(primary_k94_mirror_tta=primary, diagnostic_k94_plain={"sentinel": True},
                      harm_guard={"sentinel": True}, context={"source": "fixed"})
        with patch.object(coatnet_screen, "score_records", return_value=copy.deepcopy(common)) as score:
            actual = exp.oai_records_score("control", "candidate", root=exp.ROOT)
        score.assert_called_once_with("control", "candidate", root=exp.ROOT)
        self.assertNotIn("status", actual)
        self.assertEqual(actual["historical_weak_gate"], "REVIEW")
        self.assertEqual(actual["oai_supervised_targets"], {t: deltas[t] for t in exp.OAI_TARGETS})
        for name, value in common.items():
            self.assertEqual(actual[name], value)

    def test_screen_uses_explicit_paired_records_and_oai_callback(self):
        from training.arms import coatnet_screen
        args = types.SimpleNamespace(collection="collection", out="screen", readout="readout",
                                     verification_note="CPU reviewed")
        with patch.object(exp.bridge, "verified_arm", return_value="control"), \
             patch.object(exp, "verified_arm", return_value="candidate"), \
             patch.object(coatnet_screen, "screen_records", return_value={"status": "B_DECISION_PENDING"}) as screen:
            self.assertEqual(exp.screen(args)["status"], "B_DECISION_PENDING")
        screen.assert_called_once_with("control", "candidate", candidate="OAI", collection="collection",
            out="screen", verification_note="CPU reviewed", readout="readout", score_fn=exp.oai_records_score)

    def test_real_historical_pair_exercises_oai_diagnostic_callback(self):
        """Exercise target readout on actual AUG1 data; this is not an OAI result."""
        result = exp.oai_records_score(exp.bridge.verified_arm("CONTROL"),
                                      exp.bridge.verified_arm("AUG1"))
        self.assertEqual(result["studies"], 888)
        self.assertNotIn("status", result)
        refs = result["primary_k94_mirror_tta"]["references"]
        expected = {target: sum(refs[name]["per_target_delta"][target]
                    for name in ("jev", "consensus")) / 2 for target in exp.OAI_TARGETS}
        self.assertEqual(result["oai_supervised_targets"], expected)
        self.assertEqual(set(result["per_target_mean_delta"]), set(exp.bridge.TARGETS))
        self.assertIn("diagnostic", result["gold58"])

    def test_new_oai_run_is_incomplete_and_fails_closed(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            exp.verified_arm()


if __name__ == "__main__":
    unittest.main()
