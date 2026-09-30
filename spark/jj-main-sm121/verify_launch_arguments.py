#!/usr/bin/env python3
"""Parse the actual launcher renders without loading weights or starting engines."""
import os
import shlex
import subprocess

from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser

for model in ('qwen38-flash-next', 'glm53-flash'):
    launcher = f'/usr/local/bin/serve-{model}-jj-main-spark.sh'
    output = subprocess.check_output(['bash', launcher], text=True,
                                     env={**os.environ, 'DRY_RUN': '1'})
    lines = [line for line in output.splitlines() if '/opt/venv/bin/vllm ' in line]
    if len(lines) != 1:
        raise RuntimeError(f'Expected one command for {model}: {output}')
    command = shlex.split(lines[0][lines[0].index('/opt/venv/bin/vllm '):])
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser')
    subcommand = ServeSubcommand()
    subcommand.subparser_init(subparsers)
    parsed = parser.parse_args(command[1:])
    subcommand.validate(parsed)
    if parsed.recurrent_checkpoint_policy != 'aligned':
        raise RuntimeError(f'Wrong checkpoint policy for {model}')
    print(f'JJ-MAIN-LAUNCH-ARGUMENTS-PASS {model}', flush=True)
