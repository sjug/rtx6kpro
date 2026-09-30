# Qwen R32 restoration, September 21

User requested rollback from Karmic. Both retained R32 containers were verified
against image `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`,
their images and bind directories existed, and the Qwen endpoint had zero running
or waiting requests. Karmic was stopped with `podman stop -t 60`, kirby then
dusty. The original R32 containers were started kirby then dusty, without changes
to their configuration. Karmic containers and images were retained, not deleted.

The monitored startup completed and a real greedy non-thinking completion
returned exactly `333` with finish_reason `stop`. The model list contains only
`Qwen3.8-Flash-Next`. Both R32 containers are running and not OOM-killed.
Captured boot kernel journals contain no NV_ERR_NO_MEMORY, Xid, or killed-process
messages. Full inspections, completion response, model list, current boot logs
and kernel journals are alongside this report. This is restoration verification,
not a new performance qualification.

GLM and rusty/toby were untouched. No source/benchmark edits, image rebuild,
prune, reset, commit, or push was performed as part of the serving rollback.
