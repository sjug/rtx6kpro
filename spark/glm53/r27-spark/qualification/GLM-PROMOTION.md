# GLM R27 aligned operational default

Following the matched aligned qualification and explicit approval, R27 aligned
is the GLM operational default on sparky, buddy, rocky, and lucky. R26 remains
available as the rollback image. Qwen and DS4 were not changed.

Runner, future-launcher gate, and tooling source commit: `4a1e55b`.

## Qualified deployment identity

- Image: `ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae`
- Runner SHA-256: `f78a965525ef99d17bb1389a6c2ae48194f265f631c11bd2a797fdde0eac1d4c`
- Deployed bind-mounted launcher SHA-256: `613a257b4648aa35f074a2d695eb7a1d3cc66fba7115207a76df2a352980cfaa`
- Checkpoint: `46aaae8a82032f77100f2f03e9cc11b391df3b4d`
- Rollback image: `bb9cb676e464425acb945118bc541fd917fa7eb7f3024418f1733b70de31e8b8`

The profile remains TP4/DCP1/MTP3, BF16 draft head, FP8 KV, 1048576 context,
0.85 utilization, 4096 batched tokens, eight sequences, RoCEnante plus PyNCCL,
and LMCache disabled. Aligned is explicit in every rank's command.

## Default change, not a new serving experiment

Only the host runner was deployed. On every node the new unoverridden dry-run
render is byte-identical to the previous runner with an explicit aligned
override. The containers were not restarted and retain their qualification
start times. All four identity receipts report the same image and running state.
The deployment command also checked that the R26 image remains present.

The image and deployed launcher predate the baked aligned default. The runner
therefore supplies the flag. The source launcher and build gate now require
aligned for the next rebuild; that launcher was not deployed over the qualified
bind-mounted file, and no rebuilt image is claimed qualified here.

Receipts: [promotion directory](glm-promotion-20260906/). Numerical acceptance
and its single-boot limits remain in [the aligned report](GLM-ALIGNED-WINDOW.md).
Startup allocation warnings and non-thinking formatting caveats are unchanged.
The upstream report is a local draft, not a filed issue.
