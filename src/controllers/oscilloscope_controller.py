"""Minimal Siglent oscilloscope controller matching the standalone test script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import argparse
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pyvisa
from pyvisa import constants as visa_constants

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


@dataclass(slots=True)
class _SiglentWaveformMeta:
    endian: str
    comm_type: int
    vertical_gain: float
    vertical_offset: float
    horiz_interval: float
    horiz_offset: float
    record_length: int


class OscilloscopeController:
    """Small controller that only supports Siglent SDS oscilloscopes."""

    def __init__(self, resource_name: Optional[str] = None) -> None:
        self.resource_name = resource_name
        self.scope: Optional[pyvisa.Resource] = None
        self._time_division_seconds: Optional[float] = None
        # Public attribute used by UI code to detect model-specific commands
        self._is_siglent: Optional[bool] = None

    @property
    def is_connected(self) -> bool:
        """Check if oscilloscope is connected."""
        return self.scope is not None

    def connect(self, fast_fail: bool = False) -> bool:
        """Connect to the oscilloscope using PyVISA."""
        if self.scope:
            return True

        resources = VISAHelper.list_resources()
        if not resources:
            logger.warning("No VISA resources found")
            # In fast-fail mode simply return False quickly for health checks.
            if fast_fail:
                return False
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
        scope.write(":SYSTem:REMote")
        idn = scope.query("*IDN?").strip()
        logger.info("Oscilloscope ID: %s", idn)

        # Heuristic: mark Siglent instruments for UI-specific commands
        idn_up = idn.upper()
        if "SIGLENT" in idn_up or "SDS" in idn_up or "0X1017" in idn_up:
            self._is_siglent = True
        else:
            self._is_siglent = False

        self.scope = scope
        return True

    def disconnect(self) -> None:
        if not self.scope:
            return
        try:
            release_commands = [
                ":SYSTem:LOCal",
                "SYSTem:LOCal",
                ":SYSTem:REMote OFF",
                "SYSTem:REMote OFF",
                ":SYSTem:LOCK OFF",
                "SYSTem:LOCK OFF",
                ":SYSTem:KEY:LOCK DISable",
                "KEY:LOCK DISable",
            ]
            for cmd in release_commands:
                try:
                    self.scope.write(cmd)
                except Exception:  # noqa: BLE001 - best effort only
                    logger.debug("Scope did not accept '%s'", cmd)
            try:
                self.scope.control_ren(visa_constants.VI_GPIB_REN_DEASSERT_GTL)
            except Exception:  # noqa: BLE001 - best effort only
                logger.debug("Scope does not support REN deassert")
            time.sleep(0.3)
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to return scope to local mode")
        try:
            self.scope.close()
        finally:
            self.scope = None

    def acquire_waveform(
        self,
        channel: int = 1,
        require_trigger: bool = True,
        trigger_timeout: float = 5.0,
        post_trigger_hold: Optional[float] = None,
    ) -> Optional[SiglentWaveform]:
        if not self.scope:
            logger.error("acquire_waveform called before connect")
            return None

        scope = self.scope

        if require_trigger:
            try:
                scope.write(":SINGle")
            except Exception as exc:  # noqa: BLE001 - best effort only
                logger.debug("Failed to start single acquisition: %s", exc)

            if not self.wait_for_trigger(timeout=trigger_timeout):
                logger.warning(
                    "Timed out waiting %.1f s for trigger; reading current display.",
                    trigger_timeout,
                )
            else:
                hold_seconds = post_trigger_hold
                if hold_seconds is None and self._time_division_seconds:
                    hold_seconds = self._time_division_seconds * 10
                if hold_seconds:
                    logger.debug(
                        "Holding %.3f s after trigger to capture full record.",
                        hold_seconds,
                    )
                    time.sleep(hold_seconds)
        else:
            try:
                scope.write(":RUN")
            except Exception as exc:  # noqa: BLE001 - best effort only
                logger.debug("Failed to set RUN mode: %s", exc)

        scope.write(f":WAVeform:SOURce C{channel}")
        scope.write(":WAVeform:MODE NORMal")
        scope.write(":WAVeform:FORMat BYTE")

        try:
            scope.clear()
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to clear scope IO buffers before waveform read")
        time.sleep(0.05)

        header = scope.query_binary_values(
            ":WAVeform:PREamble?",
            datatype="B",
            container=bytearray,
        )
        meta = self._decode_preamble(header)
        if not meta:
            return None

        logger.debug(
            "Waveform meta: comm_type=%s gain=%.6e offset=%.6e dt=%.6e t0=%.6e record=%d",
            meta.comm_type,
            meta.vertical_gain,
            meta.vertical_offset,
            meta.horiz_interval,
            meta.horiz_offset,
            meta.record_length,
        )

        data_kwargs: dict[str, object] = {
            "datatype": "b" if meta.comm_type == 0 else "h",
            "container": np.array,
        }
        if meta.comm_type == 1:
            data_kwargs["is_big_endian"] = meta.endian == ">"

        samples = scope.query_binary_values(
            ":WAVeform:DATA?",
            **data_kwargs,
        )
        if samples.size == 0:
            logger.error("Scope returned no samples")
            return None

        if meta.record_length and meta.record_length != samples.size:
            logger.debug(
                "Record length mismatch: preamble reports %d points, received %d",
                meta.record_length,
                samples.size,
            )

        dtype = np.int16 if meta.comm_type == 1 else np.int8
        samples = samples.astype(dtype, copy=False)
        logger.debug("Sample codes: min=%d max=%d", int(samples.min()), int(samples.max()))

        if np.isclose(meta.horiz_interval, 0.0):
            fallback_span = None
            if self._time_division_seconds:
                fallback_span = self._time_division_seconds * 10.0
            if fallback_span and samples.size > 1:
                meta.horiz_interval = fallback_span / (samples.size - 1)
                meta.horiz_offset = 0.0
                logger.debug(
                    "Horizontal interval reported as 0; using fallback %.6e s per sample.",
                    meta.horiz_interval,
                )
            else:
                logger.warning("Horizontal interval reported as 0; cannot infer time axis.")
                return None

        voltages = (samples.astype(np.float64) - meta.vertical_offset) * meta.vertical_gain
        times = meta.horiz_offset + np.arange(samples.size, dtype=np.float64) * meta.horiz_interval
        if times[0] > times[-1]:
            times = times[::-1]
            voltages = voltages[::-1]
        return SiglentWaveform(voltages=voltages, times=times)

    def clear_persistence(self) -> None:
        if not self.scope:
            return
        try:
            self.scope.write("PESU OFF")
            self.scope.write("PESU ON,INFINITE")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to toggle persistence")

    def _send_command(self, command: str, read_response: bool = False) -> Optional[str]:
        """Send a SCPI command to the instrument.

        If ``read_response`` is True the response string is returned (stripped).
        This is a small compatibility shim used by older UI code.
        """
        if not self.scope:
            raise RuntimeError("_send_command called before connect")

        try:
            if read_response:
                # Use query for commands that return a response
                return self.scope.query(command).strip()
            else:
                self.scope.write(command)
                return None
        except Exception as e:
            logger.error("SCPI command failed '%s': %s", command, e)
            raise

    def acquire_single_waveform(self, channel: int = 1) -> Optional[np.ndarray]:
        """Acquire a single waveform and return the voltage samples as a numpy array.

        This wraps :meth:`acquire_waveform` and is provided for compatibility
        with code that expects a simple array of samples.
        """
        wf = self.acquire_waveform(channel=channel, require_trigger=False)
        if wf is None:
            return None
        return wf.voltages

    def reset_to_normal_mode(self) -> None:
        """Return the oscilloscope to a sensible normal display mode.

        Used by UI cleanup code to disable persistence and set trigger/display
        back to normal viewing.
        """
        if not self.scope:
            return
        try:
            try:
                # Normal trigger/display modes
                self.scope.write("TRMD NORM")
            except Exception:
                logger.debug("Failed to set TRMD NORM")
            try:
                # Disable persistence and other transient modes
                self.scope.write("PESU OFF")
            except Exception:
                logger.debug("Failed to disable persistence")
            try:
                # Ensure acquisition is in RUN mode
                self.scope.write(":RUN")
            except Exception:
                logger.debug("Failed to set RUN mode")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to reset oscilloscope to normal mode")

    def take_screenshot(
        self,
        file_path: str | Path,
        command: str = "SCDP",
        timeout: float = 20.0,
    ) -> Optional[Path]:
        """Capture a screen image from the scope and save to `file_path`.

        Many Siglent examples use the simple `SCDP` command which returns a
        BMP image as raw bytes. Some instruments expose other hardcopy
        commands; you can pass a different `command` if needed.

        Returns the saved :class:`pathlib.Path` on success or ``None`` on
        failure.
        """
        if not self.scope:
            logger.error("take_screenshot called before connect")
            return None

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Be conservative with timeouts for potentially large images.
        prev_timeout = getattr(self.scope, "timeout", None)
        try:
            # pyvisa timeouts are in milliseconds for many backends; use seconds
            # as a convenience but write the backend value directly if it's int.
            self.scope.timeout = int(timeout * 1000)
            try:
                # Prefer write (command may be a simple trigger like "SCDP").
                self.scope.write(command)
            except Exception:
                # If write fails, try query style (some commands are '?'-terminated).
                try:
                    raw = self.scope.query(command)
                except Exception as exc:  # noqa: BLE001 - best effort only
                    logger.error("Failed to send screenshot command '%s': %s", command, exc)
                    return None
                else:
                    # Received text response; save as bytes.
                    data = raw.encode("latin1", errors="surrogateescape")
                    with open(path, "wb") as f:
                        f.write(data)
                    logger.info("Saved screenshot to %s (query style)", path)
                    return path

            # Read raw bytes from the instrument. Many scopes send a binary block
            # (e.g. '#<digits><len><data>'). read_raw() returns the raw bytes
            # including any block header which we write straight to disk.
            try:
                data = self.scope.read_raw()
            except Exception as exc:  # noqa: BLE001 - best effort only
                logger.error("Failed to read screenshot bytes: %s", exc)
                return None

            if not data:
                logger.error("No data received for screenshot command '%s'", command)
                return None

            with open(path, "wb") as f:
                f.write(data)

            logger.info("Saved screenshot to %s", path)
            return path
        finally:
            if prev_timeout is not None:
                try:
                    self.scope.timeout = prev_timeout
                except Exception:
                    # Best-effort restore; ignore failures
                    pass

    def set_time_division(self, time_div: str) -> None:
        if not self.scope:
            return
        seconds = self._parse_time_division_seconds(time_div)
        try:
            self.scope.write(f"TDIV {time_div}")
            if seconds is not None:
                self._time_division_seconds = seconds
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Failed to set TDIV to %s", time_div)
        else:
            if seconds is None:
                logger.debug("Could not parse time division value '%s'", time_div)

    def wait_for_trigger(self, timeout: float = 5.0, poll_interval: float = 0.05) -> bool:
        if not self.scope:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = self.scope.query(":TRIGger:STATus?").strip().upper()
            except Exception as exc:  # noqa: BLE001 - best effort only
                logger.debug("Trigger status query failed: %s", exc)
                time.sleep(poll_interval)
                continue

            if status.startswith("TD") or status.startswith("STOP"):
                return True

            time.sleep(poll_interval)

        return False

    def configure_basic_view(self, channel: int = 1, time_div: str = "100MS") -> None:
        if not self.scope:
            return
        try:
            self.scope.write(f"C{channel}:TRA ON")
            self.scope.write(f"C{channel}:VDIV 0.5V")
            self.set_time_division(time_div)
            self.scope.write("TRMD NORM")
        except Exception:  # noqa: BLE001 - best effort only
            logger.debug("Basic configuration failed")

    @staticmethod
    def _parse_time_division_seconds(value: str) -> Optional[float]:
        cleaned = value.strip().upper()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Z]+)?", cleaned)
        if not match:
            return None

        magnitude = float(match.group(1))
        suffix = match.group(2) or "S"

        unit_scale = {
            "S": 1.0,
            "SEC": 1.0,
            "MS": 1e-3,
            "US": 1e-6,
            "NS": 1e-9,
            "PS": 1e-12,
        }.get(suffix)

        if unit_scale is None:
            return None

        return magnitude * unit_scale

    @staticmethod
    def _decode_preamble(header: bytearray) -> Optional[_SiglentWaveformMeta]:
        if len(header) < 200:
            logger.error("WAVeform:PREamble response too short (%d bytes)", len(header))
            return None

        comm_order_big = int.from_bytes(header[34:36], "big", signed=False)
        comm_order_little = int.from_bytes(header[34:36], "little", signed=False)

        if comm_order_big in (0, 1):
            endian = ">" if comm_order_big == 0 else "<"
        elif comm_order_little in (0, 1):
            endian = ">" if comm_order_little == 0 else "<"
        else:
            logger.error(
                "Unable to determine waveform endianness (comm_order bytes=%s)",
                header[34:36],
            )
            return None

        comm_type = struct.unpack_from(endian + "h", header, 32)[0]
        if comm_type not in (0, 1):
            logger.error("Unsupported waveform comm_type %s", comm_type)
            return None

        vertical_gain = struct.unpack_from(endian + "f", header, 156)[0]
        vertical_offset = struct.unpack_from(endian + "f", header, 160)[0]
        horiz_interval = struct.unpack_from(endian + "f", header, 176)[0]
        horiz_offset = struct.unpack_from(endian + "d", header, 180)[0]
        record_length = struct.unpack_from(endian + "I", header, 116)[0]

        return _SiglentWaveformMeta(
            endian=endian,
            comm_type=comm_type,
            vertical_gain=vertical_gain,
            vertical_offset=vertical_offset,
            horiz_interval=horiz_interval,
            horiz_offset=horiz_offset,
            record_length=record_length,
        )

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
        "--time-div",
        default="100MS",
        help="Time per division value to set (default: 100MS).",
    )
    parser.add_argument(
        "--trigger-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a trigger before falling back to the current display.",
    )
    parser.add_argument(
        "--hold-after-trigger",
        type=float,
        help="Seconds to wait after a trigger fires before reading the waveform (defaults to TDIV×10).",
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Capture immediately without arming the trigger.",
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
            controller.configure_basic_view(channel=args.channel, time_div=args.time_div)
        else:
            controller.set_time_division(args.time_div)

        waveform = controller.acquire_waveform(
            channel=args.channel,
            require_trigger=not args.no_trigger,
            trigger_timeout=args.trigger_timeout,
            post_trigger_hold=args.hold_after_trigger,
        )
        if waveform is None:
            logger.error("Scope returned no waveform data.")
            return 2

        print(f"Captured {waveform.times.size} samples on channel {args.channel}")
        voltage_min = float(waveform.voltages.min())
        voltage_max = float(waveform.voltages.max())
        voltage_ptp = float(voltage_max - voltage_min)
        print(
            f"Time span: {waveform.times[0]:.6e} s → {waveform.times[-1]:.6e} s\n"
            f"Voltage span: {voltage_min:.3e} V → {voltage_max:.3e} V (Δ={voltage_ptp:.3e} V)"
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