# Podman keep-latest-3 prune plan (2026-09-15)

Plan only — removals NOT executed. Rule: keep the serving image (running container) + 2 most recent named images per node.

| node | total | keep | remove | removable now | blocked (container) | nominal sum to free |
|---|---:|---:|---:|---:|---:|---|
| sparky | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| buddy | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| lucky | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| rocky | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| rusty | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| toby | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| dusty | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |
| kirby | 3 | 3 | 0 | 0 | 0 | 0 images, ~0.0GB nominal |

Per node: `<node>.keep.txt`, `<node>.remove.tsv`, `<node>.rmi-command.txt`. Images referenced by any container (even exited) fail plain `rmi`; those are flagged `USED BY container ...` in the remove TSV — remove the container first (`podman rm`) or use `rmi -f` deliberately.

