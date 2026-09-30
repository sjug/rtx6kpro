"""Local-only draft of the next diagnostic seam. Not an installable image kit.

This composes against the reviewed c4a51be window-capture attention source.
It does not write files, edit a manifest, build, deploy or access a GPU.
The capture helper, new lock and direct build/run approval are still required.
"""
import ast
import hashlib
import textwrap

INPUT_SHA256 = '72bee917698de5245d6731f25107f73b51cd29be249541989be8147ca6403c45'
IMPORT = 'import vllm.models.deepseek_v4_1.claude_window as _claude_window\n'
NEW_IMPORT = 'import vllm.models.deepseek_v4_1.ds41_indexer_capture as _indexer_capture\n'
ANCHOR = '''                weight_scale.scale_index_weights(
                    weights,
                    out=iw,
                    plan=self._helper_plan("index_weights"),
                )
                index_query = (iq_data, iq_scale, iw)
'''
HOOK = '''                # DIAGNOSTIC DRAFT: no projection or scaling policy change.
                if self.layer_id == 2:
                    _indexer_capture.capture(
                        self, metadata=metadata, positions=positions,
                        hidden_input=hidden_states, kv_norm=kv, q_rotated=q,
                        index_query_rotated=iq, raw_weights=weights,
                        scaled_weights=iw, projection_weight=self.indexer.weights_proj.weight,
                    )
'''


def compose(source):
    if hashlib.sha256(source.encode()).hexdigest() != INPUT_SHA256:
        raise ValueError('Expected the exact reviewed c4a51be attention source')
    if source.count(IMPORT) != 1 or source.count(ANCHOR) != 1:
        raise ValueError('Diagnostic hook seam is ambiguous or missing')
    result = source.replace(IMPORT, IMPORT + NEW_IMPORT).replace(
        ANCHOR, ANCHOR.replace('                index_query =', HOOK + '                index_query ='))
    verify_only_hook_changed(source, result)
    return result


def verify_only_hook_changed(original, instrumented):
    """Removing exactly the new import and guarded call must recover the AST."""
    before, after = ast.parse(original), ast.parse(instrumented)
    hook_ast = ast.dump(ast.parse(textwrap.dedent(HOOK)).body[0])

    class RemoveHook(ast.NodeTransformer):
        imports = 0
        calls = 0

        def visit_Import(self, node):
            if ast.dump(node) == ast.dump(ast.parse(NEW_IMPORT).body[0]):
                self.imports += 1
                return None
            return node

        def visit_If(self, node):
            if ast.dump(node) == hook_ast:
                self.calls += 1
                return None
            return self.generic_visit(node)

    remover = RemoveHook()
    cleaned = remover.visit(after)
    if (remover.imports, remover.calls) != (1, 1) or ast.dump(before) != ast.dump(cleaned):
        raise ValueError('Changes extend beyond the single guarded observation hook')
