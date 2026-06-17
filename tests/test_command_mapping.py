"""Flexible command mapping."""

import unittest

from neurobci.control.mapping import CommandMap


class TestCommandMapping(unittest.TestCase):
    def test_identity(self):
        m = CommandMap.identity(4)
        self.assertEqual(m.action_for(2), "item_2")
        self.assertEqual(m.action_for(99), "none")  # default

    def test_directional(self):
        m = CommandMap.directional()
        self.assertEqual(m.action_for(0), "up")
        self.assertEqual(m.action_for(3), "left")

    def test_to_command_carries_action_and_item(self):
        m = CommandMap.directional()
        cmd = m.to_command(1, confidence=0.8, margin=0.2)
        self.assertEqual(cmd.action, "right")
        self.assertEqual(cmd.item, 1)
        self.assertEqual(cmd.confidence, 0.8)

    def test_custom_map(self):
        m = CommandMap({0: "yes", 1: "no"}, default="abstain")
        self.assertEqual(m.action_for(0), "yes")
        self.assertEqual(m.action_for(5), "abstain")


if __name__ == "__main__":
    unittest.main()
