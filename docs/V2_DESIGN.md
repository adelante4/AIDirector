# AIDirector v2: post in, video out

Run offline with stubs: `python -m director_v2 --post sample_post.txt` (needs ffmpeg).

## Pipeline
ingest -> story -> style blend -> **bible (locked)** -> shot list -> render (keyframes, clip, critic, retry) -> assemble

Every stage writes `output/<run_id>/NN_name.json`. Reruns resume from the last checkpoint, so you can edit a
JSON file (e.g. the story) and only pay for downstream stages.

## Ideas borrowed
| From | Idea | Where |
|---|---|---|
| ArcReel / VideoMemory | Bible of characters, props, locations is generated and locked before any shot | `stages.build_bible`, `validate_bible_covers_shots` |
| VideoMemory | Asset records hold generation prompt + image path; shots reference assets by name | `contracts.Asset` |
| ViMax | First and last frame per shot; `variation_type` decides if a last frame is generated | `stages.render` |
| ViMax | Reference selection per shot (needed assets + previous frame for continuity) | `stages._refs_for` |
| ViMax | Automated QC with regeneration | critic loop in `stages.render` |
| VideoMemory | Retry on tool failure | TODO: wrap real adapters with backoff |

## Ours (not in the references)
- Input is a social post plus comments, abstracted into a story (never a literal retelling).
- Style blend over literary and film influences conditions bible, shots and prompts. Currently stubbed.

## Cost control
`Ledger` enforces a hard USD cap per run and logs cost per stage. `draft` mode is cheap, `final` raises quality.

## Next
1. Real adapters: fal.ai (video, images), Claude (LLM + vision critic).
2. Style blend retrieval (port v1 index, add films and painters, add blend weights).
3. Best-of-N keyframes with critic selection (`Config.n_candidates`, not wired yet).
4. Audio (native video audio first, TTS/music later) and crossfades.
5. Eval harness: fixed set of posts, compare models by critic score, cost, latency.
6. Port stage graph to LangGraph.
