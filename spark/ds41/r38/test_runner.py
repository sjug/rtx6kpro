"""Offline contract checks. No Podman, network, GPU or model weights required."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import contract
import disk_probe
import launch
import run_node
import verify_model


def option(command, key):
    return command[command.index("--" + key) + 1]


class RunnerTests(unittest.TestCase):
    def test_all_ranks_render(self):
        for node, (rank, address) in contract.NODES.items():
            with self.subTest(node=node):
                cfg = contract.settings({})
                cmd = contract.serve_command(cfg, node)
                self.assertEqual(option(cmd, "node-rank"), str(rank))
                self.assertEqual(option(cmd, "tensor-parallel-size"), "4")
                self.assertEqual(option(cmd, "nnodes"), "4")
                self.assertEqual(option(cmd, "decode-context-parallel-size"), "1")
                self.assertEqual(option(cmd, "distributed-executor-backend"), "mp")
                self.assertEqual(option(cmd, "master-addr"), "10.11.11.7")
                self.assertEqual("--headless" in cmd, rank != 0)
                self.assertEqual(contract.runtime_environment(cfg, node)["VLLM_HOST_IP"], address)

    def test_model_and_memory_contract(self):
        cmd = contract.serve_command(contract.settings({}), "dusty")
        self.assertEqual(option(cmd, "served-model-name"), "DeepSeek-V4.1-Flash")
        self.assertIn(contract.REVISION, cmd[4])
        self.assertEqual(option(cmd, "max-model-len"), "131072")
        self.assertEqual(option(cmd, "gpu-memory-utilization"), "0.85")
        self.assertEqual(option(cmd, "swa-block-size"), "128")
        self.assertEqual(json.loads(option(cmd, "engram-config")), {
            "cpu_offload": False, "table_memory": "disk", "disk_resident_scales": False,
            "disk_prefetch_max_tokens": 0, "projection_tp": False})
        self.assertNotIn("--kv-cache-memory-bytes", cmd)
        self.assertIn("--default-chat-template-kwargs.reasoning_effort=max", cmd)

    def test_capture_tracks_speculation(self):
        for tokens in (0, 3, 7):
            cfg = contract.settings({"DSPARK_TOKENS": str(tokens)})
            cmd = contract.serve_command(cfg, "dusty")
            self.assertEqual(option(cmd, "max-cudagraph-capture-size"), str(4 * (1 + tokens)))
            self.assertEqual("--speculative-config" in cmd, bool(tokens))
            if tokens:
                self.assertEqual(json.loads(option(cmd, "speculative-config"))["num_speculative_tokens"], tokens)

    def test_fixed_controls_reject_drift(self):
        bad = {"DCP_SIZE": "2", "TP_SIZE": "2", "NNODES": "2", "ENGRAM_TABLE_MEMORY": "ram",
               "MASTER_ADDR": "10.11.1.1", "NCCL_IB_HCA": "rocep1s0f1", "NCCL_PROTO": "LL128",
               "VLLM_ENABLE_ROCE_ALLREDUCE": "1", "LMCACHE_ENABLED": "1", "MODEL_REVISION": "main",
               "EXPECTED_IMAGE_ID": "new", "IMAGE": "latest", "NCCL_MAX_NCHANNELS": "4",
               "NCCL_LAUNCH_ORDER_IMPLICIT": "1", "EXTRA_VLLM_ARGS": "--enforce-eager",
               "KV_CACHE_MEMORY_BYTES": "123", "GPU_MEMORY_UTILIZATION": "0.90",
               "MAX_MODEL_LEN": "1048577", "MAX_NUM_SEQS": "8", "DSPARK_TOKENS": "-1"}
        for name, value in bad.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                contract.settings({name: value})

    def test_jit_warning_is_target_only_correctness_control(self):
        cfg = contract.settings({})
        cmd = contract.serve_command(cfg, "dusty")
        self.assertEqual(cmd[cmd.index("--jit-monitor-mode") + 1], "error")
        cfg = contract.settings({"DSPARK_TOKENS": "0", "JIT_MONITOR_MODE": "warn"})
        cmd = contract.serve_command(cfg, "dusty")
        self.assertEqual(cmd[cmd.index("--jit-monitor-mode") + 1], "warn")
        for env in ({"JIT_MONITOR_MODE": "warn"},
                    {"DSPARK_TOKENS": "0", "JIT_MONITOR_MODE": "off"}):
            with self.assertRaises(RuntimeError):
                contract.settings(env)

    def test_live_node_cannot_be_spoofed(self):
        for env in ({"NODE": "dusty"}, {"NODE": "kirby", "ROLE": "head"},
                    {"NODE_RANK": "0"}, {"HOST_IP": "192.168.2.1"}):
            with self.subTest(env=env), self.assertRaises(RuntimeError):
                run_node.node_for(env, "kirby")
        self.assertEqual(run_node.node_for({"DRY_RUN": "1", "NODE": "dusty"}, "workstation"), "dusty")

    def test_dry_run_has_no_external_calls_or_directory_creation(self):
        for node in contract.NODES:
            env = {"DRY_RUN": "1", "NODE": node, "CACHE": "/not-created/ds41"}
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", ["run_node.py"]), \
                 patch.object(subprocess, "run", side_effect=AssertionError("external command")), \
                 patch.object(subprocess, "check_output", side_effect=AssertionError("external command")), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("filesystem mutation")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                run_node.main()
            rendered = json.loads(output.getvalue())
            self.assertEqual(rendered["status"], "DRY_RUN")
            self.assertEqual(rendered["node"], node)

    def test_podman_mount_and_identity(self):
        cfg = contract.settings({})
        cmd, hf, _ = run_node.command({}, "dusty", cfg, "digest")
        self.assertEqual(option(cmd, "pull"), "never")
        self.assertIn(contract.IMAGE_ID, cmd)
        self.assertIn(f"{hf}:/root/.cache/huggingface:ro", cmd)
        self.assertEqual(option(cmd, "ipc"), "host")
        for forbidden in ("--privileged", "seccomp=unconfined", "--replace", "--rm", "stop", "rm"):
            self.assertNotIn(forbidden, cmd)
        self.assertIn("NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0", cmd)

    def test_cache_overlap_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hf = root / "huggingface"
            hf.mkdir()
            alias = root / "alias"
            alias.symlink_to(hf, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                run_node.command({"HF_CACHE": str(hf), "CACHE": str(alias)}, "dusty", contract.settings({}), "x")

    def test_manifest_and_tamper(self):
        self.assertEqual(len(contract.verify_kit()), 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime-files.sha256").write_text("0" * 64 + "  contract.py\n")
            (root / "contract.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                contract.verify_kit(root)

    def test_allow_all_seccomp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps({"defaultAction": "SCMP_ACT_ALLOW"}))
            with self.assertRaisesRegex(RuntimeError, "deny"):
                run_node.command({"IO_URING_SECCOMP_PROFILE": str(path)}, "dusty", contract.settings({}), "x")

    @staticmethod
    def image():
        return json.dumps([{"Id": contract.IMAGE_ID, "Architecture": "arm64", "Labels": {
            "local-inference.release.name": "jj-r38-spark-sm121", "local-inference.lmcache.default": "disabled"}}])

    def test_failed_idle_probe_is_not_idle(self):
        with patch.object(run_node.platform, "machine", return_value="aarch64"), \
             patch.object(run_node, "read", side_effect=[self.image(), subprocess.CalledProcessError(127, "podman")]):
            with self.assertRaises(subprocess.CalledProcessError):
                run_node.preflight("dusty", Path("/unused"))

    def test_running_containers_rejected(self):
        with patch.object(run_node.platform, "machine", return_value="aarch64"), \
             patch.object(run_node, "read", side_effect=[self.image(), "qwen-production"]):
            with self.assertRaisesRegex(RuntimeError, "Existing containers"):
                run_node.preflight("dusty", Path("/unused"))

    def test_bare_gpu_process_rejected(self):
        with patch.object(run_node.platform, "machine", return_value="aarch64"), \
             patch.object(run_node, "read", side_effect=[self.image(), "", "1234"]):
            with self.assertRaisesRegex(RuntimeError, "GPU processes"):
                run_node.preflight("dusty", Path("/unused"))

    def test_failed_exists_probe_rejected(self):
        for status in (0, 125, 127):
            with self.subTest(status=status), patch.object(run_node.platform, "machine", return_value="aarch64"), \
                 patch.object(run_node, "read", side_effect=[self.image(), "", ""]), \
                 patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], status)):
                with self.assertRaisesRegex(RuntimeError, "refusing replacement"):
                    run_node.preflight("dusty", Path("/unused"))

    def test_idle_memory_admission(self):
        good = "MemAvailable: 105906176 kB\nSwapTotal: 4194304 kB\nSwapFree: 4194304 kB\n"
        run_node.verify_memory(good)
        with self.assertRaisesRegex(RuntimeError, "MemAvailable"):
            run_node.verify_memory(good.replace("105906176", "1000000"))
        with self.assertRaisesRegex(RuntimeError, "swap"):
            run_node.verify_memory(good.replace("SwapFree: 4194304", "SwapFree: 1"))

    def test_both_fabric_rails_and_gid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = []
            for hca, iface, subnet in (("rocep1s0f0", "enp1s0f0np0", "10.11.11"),
                                       ("roceP2p1s0f0", "enP2p1s0f0np0", "10.11.12")):
                port = root / hca / "ports/1"
                for name, value in {"state": "4: ACTIVE", "gid_attrs/types/3": "RoCE v2",
                                    "gid_attrs/ndevs/3": iface, "gids/3": f"::ffff:{subnet}.7"}.items():
                    path = port / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value)
                answers.append(json.dumps([{"addr_info": [{"local": f"{subnet}.7"}]}]))
                answers.extend([json.dumps([{"dev": iface, "prefsrc": f"{subnet}.7"}])] * 3)
            with patch.object(run_node, "read", side_effect=answers):
                run_node.verify_fabric("dusty", root)
            (root / "roceP2p1s0f0/ports/1/gids/3").write_text("::ffff:10.11.2.1")
            with patch.object(run_node, "read", side_effect=answers), self.assertRaisesRegex(RuntimeError, "GID 3"):
                run_node.verify_fabric("dusty", root)
            bad_route = answers.copy()
            bad_route[5] = json.dumps([{"dev": "eno1", "prefsrc": "192.168.2.7"}])
            with patch.object(run_node, "read", side_effect=bad_route), self.assertRaisesRegex(RuntimeError, "switched rail"):
                run_node.verify_fabric("dusty", root)

    def test_container_drift_blocks_before_preflight(self):
        cfg = contract.settings({})
        env = contract.runtime_environment(cfg, "dusty") | {"DS41_KIT_SHA256": "good"}
        with patch.dict(os.environ, env, clear=True), patch.object(launch, "verify_kit", return_value="good"), \
             patch.object(launch, "preflight") as preflight, patch.object(os, "execv") as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            launch.main()
            preflight.assert_called_once()
            self.assertEqual(execute.call_args.args[1], contract.serve_command(cfg, "dusty"))
        for name, value in (("DS41_KIT_SHA256", "wrong"), ("NCCL_IB_GID_INDEX", "0")):
            with patch.dict(os.environ, env | {name: value}, clear=True), \
                 patch.object(launch, "verify_kit", return_value="good"), patch.object(launch, "preflight") as preflight:
                with self.assertRaises(RuntimeError):
                    launch.main()
                preflight.assert_not_called()

    def test_disk_probe_byte_comparison_and_actual_api_shape(self):
        class Native:
            corrupt = False

            def ple_reader(self, *args):
                self.geometry = args
                self.planes = {}
                return object()

            def ple_reader_add(self, reader, shard, path, offset, scale):
                self.planes[scale] = (path, offset)

            def ple_reader_run(self, reader, ids, weights, scales, count):
                for scale, output, width in ((False, weights, 256), (True, scales, 8)):
                    path, offset = self.planes[scale]
                    with open(path, "rb") as data:
                        for i, row in enumerate(ids):
                            data.seek(offset + row * width)
                            output[i * width:(i + 1) * width] = data.read(width)
                if self.corrupt:
                    weights[0] ^= 1

            def ple_reader_stats(self, reader):
                return {"lookups": 3}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-00001-of-00048.safetensors"
            path.write_bytes(bytes(range(256)) * 64)
            native = Native()
            with contextlib.redirect_stdout(io.StringIO()):
                disk_probe.verify(native, directory)
            self.assertEqual(native.geometry, (16, 16, 0, 16, 256, 8, 128, 128))
            native.corrupt = True
            with self.assertRaisesRegex(RuntimeError, "byte parity"):
                disk_probe.verify(native, directory)

    def test_model_metadata_checks_without_real_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "hub" / contract.REPO_DIR / "snapshots" / contract.REVISION
            snapshot.mkdir(parents=True)
            shards = [f"model-{i:05}.safetensors" for i in range(48)]
            config = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
                      "text_config": {"max_position_embeddings": 1048576, "num_nextn_predict_layers": 3}}
            index = {"metadata": {"total_size": 510286023000}, "weight_map": dict(zip(shards, shards))}
            files = []
            for name, value in (("config.json", config), ("model.safetensors.index.json", index)):
                data = json.dumps(value).encode()
                (snapshot / name).write_bytes(data)
                files.append({"path": name, "size": len(data),
                              "git_blob": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
            for name in shards:
                (snapshot / name).write_bytes(b"test")
                files.append({"path": name, "size": 4, "sha256": hashlib.sha256(b"test").hexdigest()})
            (root / "model-manifest.json").write_text(json.dumps({"revision": contract.REVISION, "files": files}))
            with patch.object(verify_model, "ROOT", root):
                self.assertTrue(verify_model.verify(root, True)["payload_hashes_checked"])
                (snapshot / shards[0]).write_bytes(b"bad!")
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    verify_model.verify(root, True)
                (snapshot / shards[0]).write_bytes(b"x")
                with self.assertRaisesRegex(RuntimeError, "truncated"):
                    verify_model.verify(root)
                (snapshot / shards[0]).unlink()
                blob = root / "blob"
                blob.write_bytes(b"test")
                (snapshot / shards[0]).symlink_to(blob)
                with self.assertRaisesRegex(RuntimeError, "Absolute HF link"):
                    verify_model.verify(root)
                (snapshot / shards[0]).unlink()
                (snapshot / shards[0]).symlink_to("../../../../blob")
                self.assertTrue(verify_model.verify(root, True)["payload_hashes_checked"])

    def test_deterministic_moe_passthrough_is_explicit_and_narrow(self):
        # Absent: no setting, no container variable, identical serve command.
        cfg = contract.settings({})
        self.assertNotIn("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", cfg)
        self.assertNotIn("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", contract.runtime_environment(cfg, "dusty"))
        base = contract.serve_command(cfg, "dusty")
        for value in ("0", "1"):
            cfg = contract.settings({"B12X_DYNAMIC_DETERMINISTIC_OUTPUT": value})
            self.assertEqual(cfg["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"], value)
            env = contract.runtime_environment(cfg, "dusty")
            self.assertEqual(env["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"], value)
            # Container env must carry it, and the vLLM CLI must not change.
            self.assertEqual(contract.serve_command(cfg, "dusty"), base)
            self.assertEqual(cfg["DSPARK_TOKENS"], 7)
            self.assertEqual(cfg["JIT_MONITOR_MODE"], "error")
        for value in ("", "2", "true", "yes", "on"):
            with self.assertRaisesRegex(RuntimeError, "must be 0 or 1"):
                contract.settings({"B12X_DYNAMIC_DETERMINISTIC_OUTPUT": value})
        # Existing controls are unchanged by the passthrough.
        with self.assertRaisesRegex(RuntimeError, "JIT warn"):
            contract.settings({"B12X_DYNAMIC_DETERMINISTIC_OUTPUT": "1", "JIT_MONITOR_MODE": "warn"})
        # The podman command exports it and the container-side drift check sees it.
        cfg = contract.settings({"B12X_DYNAMIC_DETERMINISTIC_OUTPUT": "1"})
        cmd, _, _ = run_node.command({}, "dusty", cfg, "x")
        self.assertIn("B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1", cmd)
        # In the container the value is re-read from the environment, echoed in
        # the printed settings receipt, and the serve command stays unchanged.
        env = contract.runtime_environment(cfg, "dusty") | {"DS41_KIT_SHA256": "good"}
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.object(launch, "verify_kit", return_value="good"), \
             patch.object(launch, "preflight"), patch.object(os, "execv") as execute, \
             contextlib.redirect_stdout(output):
            launch.main()
        self.assertEqual(json.loads(output.getvalue().splitlines()[0])["settings"]["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"], "1")
        self.assertEqual(execute.call_args.args[1], base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
