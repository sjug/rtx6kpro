#!/usr/bin/env python3
"""Diagnostic repeats of the unchanged semantic tool gate, never a pass override."""
import importlib.util
import json
from pathlib import Path

kit = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('semantic', kit.parent / 'glm53/verify-semantic-admission.py')
semantic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic)
path = kit / 'qualification/glm-allocator-20260921/tool-format-diagnostics.jsonl'
original = semantic.post_json
with path.open('x') as output:
    for repetition in range(1, 6):
        exchanges = []
        def recorded(endpoint, payload):
            response = original(endpoint, payload)
            exchanges.append({'request': payload, 'response': response})
            return response
        semantic.post_json = recorded
        try:
            result = semantic.tool_gate('http://sparky:8000/v1/chat/completions', 'GLM-5.3-Flash')
        except AssertionError as error:
            result = {'valid': False, 'error': str(error)}
        output.write(json.dumps({'repetition': repetition, 'result': result, 'exchanges': exchanges}) + '\n')
        output.flush()
        print(json.dumps({'repetition': repetition, 'result': result}), flush=True)
