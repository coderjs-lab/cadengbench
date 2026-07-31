# Runner Contracts

## Boundary

```text
frozen evalset -> CanonicalModelRequestV1/V2 -> provider adapter
               -> raw response + attempt ledger -> CandidateSubmissionV1/V2
             -> cadengbench_p_evaluator -> L0/L1/L2 evidence
```

Generation and scoring are independently repeatable. Regrading starts from the
candidate bundle and never calls a provider.

`benchmark_plan.json` is the paid-inference boundary. It binds the frozen
L0-L2 release hash, exact model specification, input profile, and both 300-item
request-manifest hashes. Preflight validates all 600 requests and their visual
assets without contacting a model endpoint. Resume rejects any changed binding.

## CanonicalModelRequestV1

Each request freezes:

| Field | Meaning |
|---|---|
| `request_id` | `<item_id>:<input_profile>:v1` |
| `dataset` | Release, item-manifest hash, and design-spec hash |
| `messages` | Exactly one system and one user message |
| `user.parts` | Authoritative text followed by profile-ordered images |
| `response_contract` | Python `submission.py` using the CadQuery v1 interface |
| `request_hash` | Canonical hash of every semantic field |

The runner accepts either a canonical `zero_to_cad/compiled` bundle or the
final release's `benchmark/consumer_ready_item_v2.json`. For the latter, it
materializes a deterministic `request_sources/<item_id>/zero_to_cad_design_source.json`
that binds the consumer record, resolved system prompt, and exact user prompt.
`design_spec_sha256` binds this immutable design source; the production item is
never modified.

For evaluator-qualified consumer records, the user message also exposes the
public parameter ID, engineering name, nominal value, unit, and role required by
the submission interface. Domains, replay schedules, measurement bindings,
tolerances, reference geometry, and hidden checks remain undisclosed.

The supported profiles are text only, text plus two canonical isometric views,
and text plus the canonical engineering four-view sheet. Image bytes live in the
request bundle and are verified against the recorded digest before transport.

## CandidateSubmissionV1

One successful candidate freezes:

| Field | Meaning |
|---|---|
| `request_hash` | Exact benchmark request answered by the model |
| `run_manifest_hash` | Exact roster, model settings, and runtime policy |
| `model` | Requested model ID plus provider-returned ID/request ID |
| `successful_attempt` | Hash of the successful append-only ledger record |
| `raw_response` | Immutable provider envelope path, size, and digest |
| `extraction` | Raw Python or one Python fence; no semantic repair |
| `artifact` | Evaluator-facing Python source path, size, and digest |
| `candidate_hash` | Canonical hash of the complete candidate record |

Packaging rejects a candidate unless its referenced ledger record exists, is a
successful record for the same item/sample/request, and binds the same raw
response digest.

## Functional-Edit V2 Contracts

`CanonicalModelRequestV2` preserves the same provenance and visual-input rules,
then adds `task_kind=functional_edit` and a source-native response contract.
BenchCAD candidates return one complete edited Python source artifact. Fusion
360 candidates return one bounded JSON patch bound to the supplied base-source
hash. `CandidateSubmissionV2` records the task kind and the exact native output
interface; no CadQuery proxy is used for Fusion 360.

The Fusion model-facing source interface is a deterministic complete index of
all replay-supported extrusion-distance and profile-circle-radius bindings. It
records exact JSON pointers, native values and units, owner feature/sketch
context, and the full native-history digest. It does not identify the selected
oracle pointer. The evaluator alone applies the patch to the full history,
checks that exactly the allowed source value changed, replays the artifact, and
compares with the hidden target evidence.

The two protocols use the same frozen 300-item roster. They are separate model
requests and evaluator runs because zero-to-CAD and editing expose different
input information and output interfaces.

## Attempt Ledger

Every attempt is appended and `fsync`ed before retry sleep. Records include
timestamps, latency, normalized terminal class, HTTP status, usage, provider
request ID, raw-response digest, retry decision, next eligible time, previous
record hash, and current record hash. A broken or truncated chain is rejected.

Durability order is:

1. Write the immutable provider envelope atomically.
2. Extract deterministically and append the attempt result.
3. Write `CandidateSubmissionV1` and `submission.py` on success.

An output-contract violation is terminal model behavior, not an infrastructure
failure. Its exact provider envelope is retained and its unique raw text is
written as `artifacts/<item_id>/response.txt` for deterministic L0 failure
scoring. A run is `run_settled` when every request has either a successful
artifact or a terminal outcome; `valid_complete_run` additionally requires an
evaluator-ready artifact for every request.

An envelope orphaned between steps 1 and 2 is recovered on resume. A successful
ledger record orphaned before step 3 is re-extracted. Neither path calls the
model again.

Completion summaries aggregate attempt status, recorded latency, and every
numeric provider-usage field. Pricing is deliberately not embedded because it
changes over time; paper cost tables can apply a dated price sheet to the frozen
token/cache usage evidence.

## Evaluator Handoff

For official `sample_count=1`, artifacts are written at:

```text
artifacts/<item_id>/submission.py
artifacts/<item_id>/submission.json
artifacts/<item_id>/response.txt
```

The JSON path is used by Fusion 360 functional edits; `response.txt` is used only
when deterministic extraction failed. This matches the evaluator batch input
convention. The evaluator reads
only the generated source and prepared evaluator cases; it has no dependency on
the runner or any provider package. `artifact_manifest.jsonl` and raw responses
remain the independent audit/regrading evidence.
