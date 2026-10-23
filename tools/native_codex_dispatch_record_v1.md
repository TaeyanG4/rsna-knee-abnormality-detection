# Native Codex archive adapter v1.0.0

B uses this only after an actual native Astra task has completed and the exact report exists. It does not run a model, log in, create reviews, or infer PASS. B copies the observed completion message into a `native-codex-completion/v1` JSON with the canonical task name, distinct native lane, model, timezone-aware observed time, exact prompt/report paths and hashes. The schema is enforced by `completion()` in the tool.

The host is explicitly **Codex app native subagent**. This is coordinator-attested evidence under the existing project archive trust boundary, not cryptographic platform proof or a successful CLI process. The adapter's SUCCESS/exit_code=0 preserves the archive ABI and denotes completion of the native task, even when the task's review verdict is FAIL. The unchanged strict gate still requires an independent lane and a correctly bound PASS report.

Commands, from the project root:

```text
python -B tools/native_codex_dispatch_record_v1.py record --completion <completion.json> --note <actual provenance>
python -B tools/native_codex_dispatch_record_v1.py verify --receipt <report.md.receipt.json>
python -B -m pytest -q tools/test_native_codex_dispatch_record_v1.py -p no:cacheprovider --basetemp=.local/pytest-tmp/native-astra-archive
```

Archive files and the report-sidecar are create-only. Partial failures retain evidence and must be inspected by B; never edit an old receipt or automatically retry. `verify` additionally checks the original prompt, completion evidence and run-log hashes, which the historical gate does not know about. B must run `verify` before native records are used by a release gate. No gate, historical stage, receipt or submit-once tool is modified by this adapter.

Production use requires a separate Astra review. Tests create visibly marked fixtures only in the supplied pytest temporary project.
