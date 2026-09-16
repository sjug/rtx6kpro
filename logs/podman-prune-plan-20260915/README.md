# Podman keep-latest-3 prune plan (2026-09-15)

Plan only — removals NOT executed. Rule: keep the serving image (running container) + 2 most recent named images per node.

| node | total | keep | remove | removable now | blocked (container) | nominal sum to free |
|---|---:|---:|---:|---:|---:|---|
| sparky | 22 | 3 | 19 | 18 | 1 | 19 images, ~586.8GB nominal |
| buddy | 22 | 3 | 19 | 18 | 1 | 19 images, ~586.8GB nominal |
| lucky | 22 | 3 | 19 | 18 | 1 | 19 images, ~586.8GB nominal |
| rocky | 22 | 3 | 19 | 18 | 1 | 19 images, ~586.8GB nominal |
| rusty | 682 | 3 | 679 | 679 | 0 | 679 images, ~16.1TB nominal |
| toby | 25 | 3 | 22 | 22 | 0 | 22 images, ~551.7GB nominal |
| dusty | 90 | 3 | 87 | 86 | 1 | 87 images, ~2.9TB nominal |
| kirby | 9 | 3 | 6 | 5 | 1 | 6 images, ~170.7GB nominal |

Per node: `<node>.keep.txt`, `<node>.remove.tsv`, `<node>.rmi-command.txt`. Images referenced by any container (even exited) fail plain `rmi`; those are flagged `USED BY container ...` in the remove TSV — remove the container first (`podman rm`) or use `rmi -f` deliberately.

