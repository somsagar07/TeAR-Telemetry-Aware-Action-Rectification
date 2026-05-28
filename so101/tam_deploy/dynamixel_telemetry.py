"""Live T/C/V telemetry from an SO-101 (Dynamixel servo) arm.

The SO-101 uses STS3215 / STS3250 / equivalent Feetech servos which expose the
same control-table layout. The class auto-detects bus topology from the IDs you
pass in and reads each servo's temperature, current and voltage on demand.

We deliberately avoid the dynamixel_sdk dependency tree (which targets ROBOTIS
servos): SO-101 ships with Feetech servos, so we use `feetech_sdk` if available,
falling back to `scservo_sdk`. The wrapper only exposes one method we need:
`read_TCV() -> (T, C, V)` where each array is shape `(n_joints,)`.

If neither SDK is installed we degrade to a `FakeDynamixelTelemetry` that
returns nominal values — useful for plumbing tests on a host without a real arm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import warnings

import numpy as np

try:
    import scservo_sdk as _sdk
    _SDK_NAME = "scservo_sdk"
except Exception:
    try:
        import feetech_sdk as _sdk
        _SDK_NAME = "feetech_sdk"
    except Exception:
        _sdk = None
        _SDK_NAME = None


# Feetech STS-series control-table register addresses.
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63
ADDR_PRESENT_CURRENT = 69


@dataclass
class DynamixelTelemetry:
    """Reads live T/C/V from an SO-101 over the Feetech serial bus.

    Parameters
    ----------
    port :
        Serial device path, e.g. "/dev/ttyACM0" or "COM7".
    baud :
        Bus baudrate. Default 1_000_000 matches Feetech factory firmware.
    ids :
        Servo IDs for the five arm joints (gripper excluded). Ordered shoulder→wrist.
    rated_current_a :
        Per-joint rated current in amps used to NORMALIZE the live reading
        before TAM consumes it. Most SO-101 builds use mixed STS3215 / STS3250
        with rated currents of 1.5–4.0 A.
    rated_voltage_v :
        Per-joint rated supply voltage. Defaults to 7.4 V (2S LiPo).
    """
    port: str
    baud: int = 1_000_000
    ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    rated_current_a: List[float] = field(
        default_factory=lambda: [3.0, 3.0, 2.0, 2.0, 1.5])
    rated_voltage_v: float = 7.4

    def __post_init__(self):
        if _sdk is None:
            warnings.warn(
                "Neither scservo_sdk nor feetech_sdk found. "
                "DynamixelTelemetry will return nominal T/C/V. "
                "Install one of: pip install scservo_sdk feetech-sdk",
                RuntimeWarning,
            )
            self._port_handler = None
            return
        self._port_handler = _sdk.PortHandler(self.port)
        self._packet = _sdk.PacketHandler(0)  # Feetech proto v0
        if not self._port_handler.openPort():
            raise IOError(f"Could not open Feetech bus at {self.port}")
        if not self._port_handler.setBaudRate(self.baud):
            raise IOError(f"Could not set baud {self.baud} on {self.port}")

    def _read_byte(self, sid: int, addr: int) -> int:
        v, comm, err = self._packet.read1ByteTxRx(self._port_handler, sid, addr)
        if comm != _sdk.COMM_SUCCESS or err != 0:
            return -1
        return int(v)

    def _read_word(self, sid: int, addr: int) -> int:
        v, comm, err = self._packet.read2ByteTxRx(self._port_handler, sid, addr)
        if comm != _sdk.COMM_SUCCESS or err != 0:
            return -1
        return int(v)

    def read_TCV(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (T_celsius, C_normalized, V_normalized).

        All arrays are shape (len(ids),). C and V are normalized to the
        rated values so they live in the same numerical range the TAM was
        trained on: C ∈ [0, 1.2], V ∈ [0, 1.0].

        Nominal (everything healthy) returns:
            T ≈ 30°C,  C ≈ 0.2,  V ≈ 1.0.
        """
        n = len(self.ids)
        if self._port_handler is None:
            T = np.full(n, 30.0, dtype=np.float32)
            C = np.full(n, 0.2, dtype=np.float32)
            V = np.full(n, 1.0, dtype=np.float32)
            return T, C, V

        T = np.zeros(n, dtype=np.float32)
        C = np.zeros(n, dtype=np.float32)
        V = np.zeros(n, dtype=np.float32)
        for k, sid in enumerate(self.ids):
            t_raw = self._read_byte(sid, ADDR_PRESENT_TEMP)        # °C
            c_raw = self._read_word(sid, ADDR_PRESENT_CURRENT)     # 6.5 mA units
            v_raw = self._read_byte(sid, ADDR_PRESENT_VOLTAGE)     # 0.1 V units
            T[k] = float(t_raw) if t_raw >= 0 else 30.0
            cur_a = (float(c_raw) * 0.0065) if c_raw >= 0 else 0.0
            volt_v = (float(v_raw) * 0.1) if v_raw >= 0 else self.rated_voltage_v
            C[k] = cur_a / max(self.rated_current_a[k], 1e-6)
            V[k] = volt_v / max(self.rated_voltage_v, 1e-6)
        return T, C, V

    def close(self):
        if self._port_handler is not None:
            self._port_handler.closePort()


def fake_cool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic telemetry that lies in the cool regime — useful for plumbing
    tests. TAM's structural gate is exactly 0 here, so the output should be
    bit-identical to the base action."""
    n = 5
    return (np.full(n, 30.0, np.float32),
            np.full(n, 0.2, np.float32),
            np.full(n, 1.0, np.float32))


def fake_hot() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic telemetry in the hot regime — exercises the TAM correction
    path. Use to verify the corrected action differs from the base."""
    n = 5
    return (np.full(n, 65.0, np.float32),
            np.full(n, 0.9, np.float32),
            np.full(n, 0.6, np.float32))
