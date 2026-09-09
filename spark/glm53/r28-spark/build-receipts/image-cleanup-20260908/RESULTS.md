# Image-only cleanup, 2026-09-08

Authorized scope: obsolete Podman images and unused intermediates on dusty and
kirby. No serving launch, container deletion, volume cleanup, cache clearing,
archive deletion, or source changes.

| Host | Unique image IDs before | After | Removed | Podman image storage before | After | Filesystem used-byte reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dusty | 2221 | 926 | 1295 | 341.2 GB | 126.8 GB | 217577218048 |
| kirby | 65 | 6 | 59 | 266 GB | 93.24 GB | 175612612608 |

Counts are deduplicated IDs from saved images-before.json and images-after.json,
not tag rows. Filesystem changes include incidental host activity; approximately
393 GB was recovered across both hosts. These immediately-before receipts take
precedence over earlier discovery snapshots.

All 38 dusty and 35 kirby explicitly selected obsolete image IDs are absent.
R28 cd93d80b3f95, R27 ef669fa1cde3, R26 bb9cb676e464, corrected GG r34
276f00868134, and corrected GG r33 69f83ec7efff retain their original identities
and names. Other retained named dependencies also passed the same check.

Dusty's initial image-removal pass stopped when automatic ancestor cleanup met
an external build-container reference. The requested obsolete image itself had
been removed. A bounded resume processed the remaining explicit names with
--no-prune, then ran ordinary image prune without --all, --external, or
--build-cache. Prune returned 125 after two protected build-container references;
these were retained, never force-removed. Final verification passed separately.
Kirby's full cleanup and verification exited zero.

All 316 dusty external build containers and their names/image references are
unchanged. Volumes are unchanged. Build dependencies and their ancestor images
explain the larger retained store on dusty. HF/model caches, JIT caches, build
directories, image archives, and benchmark results were untouched.

Inventories record identities, not backups of deleted images. Recovering a
deleted local-only image requires an existing archive or a rebuild; registry
images can be pulled again. No new image archives were created for cleanup.

Per-host JSON inventories, plans, storage and filesystem measurements, and logs
are saved in the dusty/ and kirby/ directories beside this report and on each
node under /home/jugs/git/bld-jj-r28-spark/qualification/image-cleanup-20260908/.
