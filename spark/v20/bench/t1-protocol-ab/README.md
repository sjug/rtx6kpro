# T1 NCCL protocol evidence

Date: 2026-08-22

These files preserve the rank-0 NCCL tuner evidence from the LL128 arm of the
DS4 r18p protocol A/C/A on `dusty` and `kirby`:

- `arm2-tuner-lines.txt` contains the selected tuning decisions and channel
  initialization lines.
- `arm2-tuner-matrix.txt` contains the corresponding enabled algorithm and
  protocol matrix.

The complete interpretation is recorded in
[`../ds4-c4-boot-distribution-20260821/NOTES.md`](../ds4-c4-boot-distribution-20260821/NOTES.md).
LL128 was enabled for the candidate arm but was not selected for the relevant
collective sizes, and the A/C/A did not establish a performance benefit. These
receipts therefore support rejection of the LL128 launcher change; they are
not a production configuration contract.
