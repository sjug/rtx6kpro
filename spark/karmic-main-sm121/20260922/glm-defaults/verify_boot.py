#!/usr/bin/env python3
"""Require the actual entrypoint's resolved command and policy on every rank."""
import hashlib
import json
import re
import sys
from pathlib import Path
import experiment

ROOT = Path(__file__).resolve().parent

def check_selective_mxfp8(boot_text):
    summaries = re.findall(r'Quantized (\d+) layers of types: ([^\n]*)', boot_text)
    nonempty = [(int(count), text) for count, text in summaries if int(count)]
    # Target and draft creation can log the same accumulated config twice.
    # Repeated identical declarations are allowed; changed/expanded sets are not.
    if (not nonempty or any(row != nonempty[0] for row in nonempty)
            or nonempty[0][0] != len(experiment.MXFP8_TARGETS)):
        raise SystemExit('Selective MXFP8 layer-count mismatch')
    entries = re.findall(r'([^;]+?): (\d+) \(from targets: ([^,]+), ([^)]+)\)', nonempty[0][1])
    observed = {target: quant for _, count, target, quant in entries if count == '1'}
    if len(entries) != len(experiment.MXFP8_TARGETS) or observed != experiment.MXFP8_TARGETS:
        raise SystemExit('Selective MXFP8 target-set mismatch')
    if 'Using B12xMxfp8LinearKernel for MXFP8 GEMM' not in boot_text:
        raise SystemExit('Selective MXFP8 did not select the B12X kernel')

def check_head(boot_text, environment):
    copy = 'Using a draft-only NVFP4 GLM MTP vocabulary head copy' in boot_text
    shared = 'Quantizing LM head shards to NVFP4 with BF16 activations.' in boot_text
    expected_shared = environment.get('VLLM_MTP_NVFP4_LM_HEAD', '1') == '1'
    expected_copy = not expected_shared and environment['VLLM_GLM53_MTP_DRAFT_HEAD'] == 'nvfp4'
    if (shared, copy) != (expected_shared, expected_copy):
        raise SystemExit('Draft-head engagement mismatch')

def main():
    arm = sys.argv[2] if len(sys.argv) == 3 else 'defaults'
    lock, arm_record, effective = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), arm)
    digest = hashlib.sha256((ROOT / 'profile.lock.json').read_bytes()).hexdigest()
    for rank, node in enumerate(('sparky', 'buddy', 'rocky', 'lucky')):
        records = []
        drivers = []
        experiments = []
        boot_text = (Path(sys.argv[1]) / (node + '-boot.log')).read_text()
        prefetch = '[l2_prefetch] CuTe kernel ready' in boot_text
        check_head(boot_text, lock['environment'])
        if arm == 'no-prefetch-mxfp8-output-a16':
            check_selective_mxfp8(boot_text)
        if prefetch != (lock['environment']['VLLM_GLM53_L2_PREFETCH'] == '1'):
            raise SystemExit('L2 prefetch engagement mismatch on ' + node)
        if rank == 0 and ('GLM speculative KDA uses B12X checkpoint recovery' in boot_text) != (arm != 'no-replayssm'):
            raise SystemExit('RecoverSSM engagement mismatch on head')
        for line in boot_text.splitlines():
            if line.startswith('{"argv":'):
                records.append(json.loads(line))
            if line.startswith('{"cuda_driver_api":'):
                drivers.append(json.loads(line))
            if line.startswith('{"experiment":'):
                experiments.append(json.loads(line))
        if experiments != [{'experiment': arm_record, 'effective_profile_sha256': effective}]:
            raise SystemExit('Experiment identity mismatch on ' + node)
        if len(drivers) != 1 or drivers[0]['cuda_driver_api'] != 13040 or drivers[0]['libcuda_mappings'] != ['/usr/local/cuda-13.4/compat/lib.real/libcuda.so.615.65.02']:
            raise SystemExit('CUDA driver identity mismatch on ' + node)
        expected = lock['argv'] + ['--nnodes', '4', '--node-rank', str(rank),
                                  '--master-addr', '10.11.11.1', '--master-port', '25000']
        if rank:
            expected.append('--headless')
        if len(records) != 1 or records[0] != {
            'argv': expected, 'profile_sha256': digest,
            'policy_environment': lock['environment'],
        }:
            raise SystemExit('Boot command/environment mismatch on ' + node)
    print('BOOT-CONTRACT-PASS: four resolved commands and policy environments')

if __name__ == '__main__':
    main()
