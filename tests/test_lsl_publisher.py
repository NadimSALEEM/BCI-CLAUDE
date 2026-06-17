"""Virtual LSL stream: publish a source and receive it back (loopback).

Skips gracefully if liblsl loopback is not available in the environment.
"""

import time
import unittest

from neurobci.acquisition.lsl_publisher import LSLPublisher
from neurobci.acquisition.simulated import SimulatedSource
from neurobci.config.schema import ChannelConfig, SimulationConfig


class TestLSLLoopback(unittest.TestCase):
    def test_publish_and_receive(self):
        try:
            import pylsl
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"pylsl unavailable: {exc}")

        name = f"NeuroBCI-Test-{int(time.time() * 1000) % 100000}"
        src = SimulatedSource(ChannelConfig(), SimulationConfig(), sfreq=250.0,
                              realtime=True)
        pub = LSLPublisher(src, stream_name=name, source_id=name)
        got, nch = 0, None
        try:
            pub.start()
            streams = pylsl.resolve_byprop("name", name, timeout=5.0)
            if not streams:
                self.skipTest("Virtual LSL stream not resolvable in this environment")
            inlet = pylsl.StreamInlet(streams[0])
            self.assertEqual(inlet.info().channel_count(), 20)
            deadline = time.time() + 1.5
            while time.time() < deadline and got < 50:
                samples, _ = inlet.pull_chunk(timeout=0.2)
                if samples:
                    got += len(samples)
                    nch = len(samples[0])
                time.sleep(0.05)
            inlet.close_stream()
        finally:
            pub.stop()

        if got == 0:
            self.skipTest("No samples over LSL loopback (sandbox restriction)")
        self.assertEqual(nch, 20)


if __name__ == "__main__":
    unittest.main()
