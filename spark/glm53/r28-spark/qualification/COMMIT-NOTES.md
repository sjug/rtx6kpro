# R28 artifact preservation

The source commit preserves the build recipe, source locks, patches, launchers
and executed qualification tools. The separate receipt commit preserves build
attempts and Qwen/GLM deployment and qualification evidence.

Image cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
was built before these commits with explicit dirty provenance. Its baked recipe
commit remains 783ad369b1b8650df2e5448dce0f0a9f14236503; the final build receipt's
recipe manifest and source/native/image identities identify what was built.
Do not equate the later whole artifact directory with that build manifest:
qualification scripts and records were added after the image was produced.

Before committing, eight build-contract tests and both runner suites were
rerun. Three newer qualification helpers needed the same optimized-Python
refusal as the existing verification entrypoints. Adding those guards made
the suite pass, without changing runtime/image content or existing results.
Shell syntax checks passed. Whitespace diagnostics in frozen upstream patch
files are retained deliberately to preserve their pinned bytes.

Container archives, the reproducible LMCache bundle, composition scratch and
Python caches remain ignored on persistent disk. Build receipts are explicitly
included despite their generated-directory ignore rule. Standard benchmark
JSONs remain in the existing private benchmark results repository, with paths
and digests in the qualification records. No raw result was rewritten.
