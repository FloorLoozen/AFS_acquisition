"""Minimal Siglent oscilloscope controller matching the standalone test script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvisa

if __package__ in (None, ""):
    # Allow running this file directly: add project root so `src` imports resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.visa_helper import VISAHelper

logger = get_logger("oscilloscope_controller")


@dataclass(slots=True)
class SiglentWaveform:
    voltages: np.ndarray
    times: np.ndarray


class OscilloscopeController:
    """Small controller that only supports Siglent SDS oscilloscopes."""

    def __init__(self, resource_name: Optional[str] = None) -> None:
        self.resource_name = resource_name
        self.scope: Optional[pyvisa.Resource] = None

    def connect(self) -> bool:
        """Connect to the oscilloscope using PyVISA."""
        if self.scope:
            return True

        resources = VISAHelper.list_resources()
        if not resources:
            logger.warning("No VISA resources found")
            return False

        target = self.resource_name
        if not target:
            for name in resources:
                upper = name.upper()
                if "USB" in upper and ("SIGLENT" in upper or "0X1017" in upper):
                    target = name
                    break
            if not target:
                target = resources[0]

        logger.info("Connecting to oscilloscope %s", target)
        scope = VISAHelper.open_resource(target)
        if not scope:
            logger.error("Failed to open VISA resource %s", target)
            return False

        scope.timeout = 20000
        scope.chunk_size = 262144
        scope.read_termination = "\n"
        scope.write_termination = "\n"
        scope.query_delay = max(getattr(scope, "query_delay", 0.0), 0.1)
        logger.info("Oscilloscope ID: %s", scope.query("*IDN?").strip())

        self.scope = scope
        return True

    def disconnect(self) -> None:
        if not self.scope:
            return
        try:
            self.scope.write("SYSTem:LOCal")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to return scope to local mode")
        try:
            self.scope.close()
        finally:
            self.scope = None

    def acquire_waveform(self, channel: int = 1) -> Optional[SiglentWaveform]:
        if not self.scope:
            logger.error("acquire_waveform called before connect")
            return None

        scope = self.scope
        scope.write(f":WAVeform:SOURce C{channel}")
        scope.write(":WAVeform:MODE NORMal")
        scope.write(":WAVeform:FORMat BYTE")

        header = scope.query_binary_values(
            ":WAVeform:PREamble?",
            datatype="B",
            container=bytearray,
        )
        if len(header) < 200:
            logger.error("WAVeform:PREamble response too short (%d bytes)", len(header))
            return None

        def _unpack(fmt: str, offset: int) -> float:
            size = np.dtype(fmt).itemsize
            return np.frombuffer(header[offset : offset + size], fmt)[0]

        comm_order = int.from_bytes(header[34:36], "big")
        endian = ">" if comm_order == 0 else "<"
        vertical_gain = float(_unpack(endian + "f", 156))
        vertical_offset = float(_unpack(endian + "f", 160))
        horiz_interval = float(_unpack(endian + "f", 176))
        horiz_offset = float(_unpack(endian + "d", 180))

        samples = scope.query_binary_values(
            ":WAVeform:DATA?",
            datatype="b",
            container=np.array,
        )
        if samples.size == 0:
            logger.error("Scope returned no samples")
            return None

        voltages = (samples.astype(np.float64) - vertical_offset) * vertical_gain
        times = horiz_offset + np.arange(samples.size, dtype=np.float64) * horiz_interval
        return SiglentWaveform(voltages=voltages, times=times)

    def clear_persistence(self) -> None:
        if not self.scope:
            return
        try:
            self.scope.write("PESU OFF")
            self.scope.write("PESU ON,INFINITE")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to toggle persistence")

    def configure_basic_view(self, channel: int = 1) -> None:
        if not self.scope:
            return
        try:
            self.scope.write(f"C{channel}:TRA ON")
            self.scope.write(f"C{channel}:VDIV 0.5V")
            self.scope.write("TDIV 100US")
            self.scope.write("TRMD AUTO")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Basic configuration failed")


_OSCILLOSCOPE: Optional[OscilloscopeController] = None


def get_oscilloscope_controller() -> OscilloscopeController:
    global _OSCILLOSCOPE
    if _OSCILLOSCOPE is None:
        _OSCILLOSCOPE = OscilloscopeController()
    return _OSCILLOSCOPE


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to a Siglent oscilloscope and display the current waveform.",
    )
    parser.add_argument(
        "--resource",
        help="Explicit VISA resource string. If omitted the first Siglent USB instrument is used.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Oscilloscope channel to capture (default: 1).",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip the basic display/trigger configuration step.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not open a matplotlib preview window; still prints waveform stats.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_cli_parser().parse_args(argv)

    controller = OscilloscopeController(resource_name=args.resource)
    if not controller.connect():
        logger.error("Could not connect to an oscilloscope. Check the USB cable and VISA drivers.")
        return 1

    try:
        if not args.no_config:
            controller.configure_basic_view(channel=args.channel)

        waveform = controller.acquire_waveform(channel=args.channel)
        if waveform is None:
            logger.error("Scope returned no waveform data.")
            return 2

        print(f"Captured {waveform.times.size} samples on channel {args.channel}")
        print(
            f"Time span: {waveform.times[0]:.6e} s → {waveform.times[-1]:.6e} s\n"
            f"Voltage span: {waveform.voltages.min():.3f} V → {waveform.voltages.max():.3f} V"
        )

        if not args.no_plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(waveform.times, waveform.voltages)
            plt.title("Siglent SDS waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return 0

    finally:
        controller.disconnect()


if __name__ == "__main__":
    sys.exit(main())