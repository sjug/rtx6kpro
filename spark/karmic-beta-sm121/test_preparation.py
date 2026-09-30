import ast
import unittest
from prepare_overlay import adapt_arches, adapt_head, replace_once


class PreparationTests(unittest.TestCase):
    def test_rejects_missing_or_repeated_input(self):
        for text in ("", "aa"):
            with self.assertRaises(ValueError):
                replace_once(text, "a", "b")

    def test_arches_only_extend_two_lists(self):
        before = ('"7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.7;11.0;12.0"\n'
                  '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"\n')
        after = adapt_arches(before)
        self.assertEqual(after.replace(";12.1", ""), before)
        self.assertEqual(after.count(";12.1"), 2)
        with self.assertRaises(ValueError):
            adapt_arches(after)

    def test_head_gate_and_reduction_unchanged(self):
        reduction = 'def scale(w):\n    return w.split(4 * 1024 * 1024)\n'
        before = reduction + ('def check(major, minor):\n'
                              '    if (major, minor) != (12, 0):\n'
                              '        raise ValueError(f"12.0; got {major}.{minor}")\n')
        after = adapt_head(before)
        self.assertTrue(after.startswith(reduction))
        namespace = {}
        exec(compile(ast.parse(after), "gate", "exec"), namespace)
        for capability in ((12, 0), (12, 1)):
            namespace["check"](*capability)
        for capability in ((11, 0), (13, 0)):
            with self.assertRaises(ValueError):
                namespace["check"](*capability)
        with self.assertRaises(ValueError):
            adapt_head(after)


if __name__ == "__main__":
    unittest.main()
