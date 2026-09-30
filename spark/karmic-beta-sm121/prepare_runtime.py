#!/usr/bin/env python3
"""Export the frozen upstream runtime inputs without changing source checkouts."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jugs/git/blackwell-llm-docker')
COMMIT = '8c5aa7f828689e6385fa309acd0cce168fb38ec7'
FILES = [
    'tools/jovian_wheel_runtime/qwen38-runtime.in',
    'tools/jovian_wheel_runtime/qwen38-runtime.lock',
    'tools/jovian_wheel_runtime/ngc-python-patches.json',
    'tools/jovian_wheel_runtime/ngc-venv-python-patches.json',
    'recipes/glm53/install_dependency_python_patches.py',
    'recipes/glm53/torch-schema-enumeration.patch',
    'recipes/glm53/cutlass-sentinel-identity.patch',
]


def main():
    manifest = {'recipe_commit': COMMIT, 'files': {}}
    for name in FILES:
        data = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{name}'])
        target = ROOT / 'runtime-inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest['files'][name] = hashlib.sha256(data).hexdigest()
    # Same exact application versions as upstream. Resolve ARM wheel hashes,
    # not the x86-only torch-c-dlpack hash in the upstream overlay lock.
    requirements = (ROOT / 'runtime-inputs/tools/jovian_wheel_runtime/qwen38-runtime.in').read_text()
    requirements += '\nquack-kernels==0.6.4\nsetuptools==80.9.0\n'
    (ROOT / 'runtime-arm64.in').write_text(requirements)
    (ROOT / 'runtime-inputs/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    old = ROOT.parent / 'glm53/r38-spark/launchers'
    launchers = ROOT / 'launchers'
    launchers.mkdir(exist_ok=True)
    origins = {}
    for model in ('qwen38-flash-next', 'glm53-flash'):
        name = f'serve-{model}-jj-r38-spark.sh'
        original = (old / name).read_bytes()
        text = original.decode().replace('JJ r38', 'Karmic beta').replace('R38 verification', 'Karmic verification')
        if model == 'glm53-flash':
            text = text.replace('serve-glm53-flash-jj-r38-spark.sh', 'serve-glm53-flash-karmic-spark.sh')
            text = text.replace('/opt/jovian-judgement/b12x-roce', '/opt/b12x-roce-cache')
        target = launchers / name.replace('jj-r38', 'karmic')
        target.write_text(text)
        target.chmod(0o755)
        origins[target.name] = {'source': str((old / name).relative_to(ROOT.parent.parent)),
                                'source_sha256': hashlib.sha256(original).hexdigest(),
                                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (launchers / 'manifest.json').write_text(json.dumps(origins, indent=2) + '\n')
    runner = (old.parent / 'run-qwen38-flash-next-jj-r38-spark-tp2-node.sh').read_text()
    start = runner.index("jq -e '\n")
    end = runner.index('\n\nmkdir -p', start)
    runner = runner[:start] + '''jq -e '
  ."vllm.source-tree" == "02a457e2e933d0fb20d5786835110546acf7d8f2" and
  ."b12x.source-tree" == "9f20c0e6b9d49a42bc19f45429ac44758cf5fe76" and
  ."lmcache.default" == "disabled" and
  ."local-inference.qwen.launcher.sha256" == "QWEN_SHA_PLACEHOLDER"
' <<<"${labels}" >/dev/null || { echo 'Karmic provenance mismatch' >&2; exit 78; }
''' + runner[end:]
    runner = runner.replace('QWEN_SHA_PLACEHOLDER', origins['serve-qwen38-flash-next-karmic-spark.sh']['sha256'])
    runner = runner.replace('jj-r38', 'karmic').replace('R38', 'Karmic').replace('r38 image', 'Karmic image')
    (ROOT / 'run-qwen-tp2-node.sh').write_text(runner)
    glm_runner = (old.parent / 'run-glm53-flash-jj-r38-spark-tp4-node.sh').read_text()
    start = glm_runner.index("jq -e '\n")
    end = glm_runner.index('\n\nif [[ -n "${GLM_LAUNCHER_FILE', start)
    glm_runner = glm_runner[:start] + '''jq -e '
  ."vllm.source-tree" == "02a457e2e933d0fb20d5786835110546acf7d8f2" and
  ."b12x.source-tree" == "9f20c0e6b9d49a42bc19f45429ac44758cf5fe76" and
  ."lmcache.default" == "disabled" and
  ."local-inference.glm.launcher.sha256" == "GLM_SHA_PLACEHOLDER"
' <<<"${labels}" >/dev/null || { echo 'Karmic provenance mismatch' >&2; exit 78; }
''' + glm_runner[end:]
    glm_runner = glm_runner.replace('GLM_SHA_PLACEHOLDER', origins['serve-glm53-flash-karmic-spark.sh']['sha256'])
    glm_runner = glm_runner.replace('jj-r38', 'karmic').replace('JJ r38', 'Karmic beta').replace('R38', 'Karmic')
    glm_runner = glm_runner.replace('/opt/jovian-judgement/b12x-roce', '/opt/b12x-roce-cache')
    glm_runner = glm_runner.replace('local-inference.launcher.glm.sha256', 'local-inference.glm.launcher.sha256')
    (ROOT / 'run-glm-tp4-node.sh').write_text(glm_runner)
    ds4_old = ROOT.parent / 'ds4-vision/r38'
    ds4 = ROOT / 'ds4-vision'
    ds4.mkdir(exist_ok=True)
    ds4_source = subprocess.check_output(['git', '-C', '/home/jugs/git/vllm', 'show',
        '57a80980bbf4b40398de7ed851b23e55a3a4c50e:serve-ds4-flash.sh'])
    new_image = 'f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233'
    for name in ('run-node.sh', 'launch-in-container.sh', 'runtime-preflight.py', 'verify-model.py'):
        text = (ds4_old / name).read_text()
        text = text.replace('ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5', new_image)
        text = text.replace('3542a3f6663503dc8697c85e27e5a0a67b80c68c9ffc76143a39403f2bf0d18c', hashlib.sha256(ds4_source).hexdigest())
        text = text.replace('/home/jugs/git/ds4-vision-r38', '/home/jugs/git/ds4-vision-r38/karmic-beta-sm121')
        text = text.replace('jj-r38', 'karmic')
        (ds4 / name).write_text(text)
    (ds4 / 'receipts').mkdir(exist_ok=True)
    (ds4 / 'receipts/model-manifest.json').write_bytes((ds4_old / 'receipts/model-manifest.json').read_bytes())
    (ds4 / 'runtime-files.sha256').write_text(''.join(
        f'{hashlib.sha256((ds4 / name).read_bytes()).hexdigest()}  {name}\n'
        for name in ('run-node.sh', 'launch-in-container.sh', 'runtime-preflight.py', 'verify-model.py')))
    tests = ROOT / 'tests'
    tests.mkdir(exist_ok=True)
    old_tests = old.parent / 'tests'
    for name in ('run_r38_regressions.py', 'run_required.py', 'flashkda_counts.py', 'verify_glm53_nvfp4_draft_head_sm121.py'):
        text = (old_tests / name).read_text()
        if name == 'run_r38_regressions.py':
            text = text.replace('test_b12x_moe_warmup_runs_each_planner_regime_once",\n        1, None,',
                                'test_b12x_moe_prefill_capacity_and_exact_decode_reuse",\n        6, None,')
            text = text.replace('plugins=[gate]', 'plugins=[gate, __import__("workspace_fixture")]')
        if name == 'verify_glm53_nvfp4_draft_head_sm121.py':
            # Karmic keeps the capability check inline, unlike the R38 overlay.
            text = text.replace('    supports_nvfp4_draft_head,\n', '')
            text = text.replace('    assert supports_nvfp4_draft_head(capability)\n', '')
        (tests / name).write_text(text.replace('R38', 'Karmic'))


if __name__ == '__main__':
    main()
