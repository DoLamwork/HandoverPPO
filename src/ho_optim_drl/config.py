"""Configuration file for the handover environment."""

from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for trace processing and the handover protocol.
    """

    # Data parameters
    delta_t_ms: int = 10  # Time step in ms (delta t)

    clip_sinr: bool = True  # Clip SINR values
    sinr_upper_clip: float = 10.0  # Upper clipping value for SINR in dB
    sinr_lower_clip: float = -10.0  # Lower clipping value for SINR in dB

    l3_k: int = 16  # L3 filter coefficient
    l3_filter_w: float = 1 / (2 ** (l3_k / 4))  # L3 filter weight

    # Network
    fc: float = 2.1e9  # Carrier frequency in Hz
    bw: float = 10e6  # bandwidth in Hz

    # Handover environment parameters
    # SINR proxies for the lower-layer Qin/Qout indications. 3GPP defines
    # Qin/Qout from hypothetical PDCCH BLER, not as universal SINR thresholds.
    q_in_db: float = -6.0
    q_out_db: float = -8.0
    t_ho_prep_ms: int = 50  # Simulation assumption
    t_ho_exec_ms: int = 40  # Simulation assumption
    t_mts_ms: int = 1_000  # Ping-pong evaluation window
    t_rlfr_ms: int = 200  # Simplified RLF recovery delay
    t_t310_ms: int = 1_000
    n310: int = 10
    n311: int = 3

    # 3GPP parameters
    a3_hys: float = 1.0  # Hysteresis for A3
    a3_ttt_ms: int = 40
    a3_off: float = 1.0

    baseline_permit_ho_prep_abort: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def _duration_steps(self, duration_ms: int, name: str) -> int:
        if duration_ms < 0:
            raise ValueError(f"{name} must be non-negative.")
        if duration_ms % self.delta_t_ms:
            raise ValueError(
                f"{name}={duration_ms} ms is not divisible by "
                f"delta_t_ms={self.delta_t_ms} ms."
            )
        return duration_ms // self.delta_t_ms

    @property
    def t_ho_prep(self) -> int:
        return self._duration_steps(self.t_ho_prep_ms, "t_ho_prep_ms")

    @property
    def t_ho_exec(self) -> int:
        return self._duration_steps(self.t_ho_exec_ms, "t_ho_exec_ms")

    @property
    def t_mts(self) -> int:
        return self._duration_steps(self.t_mts_ms, "t_mts_ms")

    @property
    def t_rlfr(self) -> int:
        return self._duration_steps(self.t_rlfr_ms, "t_rlfr_ms")

    @property
    def t_t310(self) -> int:
        return self._duration_steps(self.t_t310_ms, "t_t310_ms")

    @property
    def a3_ttt(self) -> int:
        return self._duration_steps(self.a3_ttt_ms, "a3_ttt_ms")

    def validate(self) -> None:
        if self.delta_t_ms <= 0:
            raise ValueError("delta_t_ms must be positive.")
        if self.q_out_db >= self.q_in_db:
            raise ValueError("q_out_db must be below q_in_db.")
        if self.a3_ttt_ms not in {
            0,
            40,
            64,
            80,
            100,
            128,
            160,
            256,
            320,
            480,
            512,
            640,
            1024,
            1280,
            2560,
            5120,
        }:
            raise ValueError("a3_ttt_ms is not a TS 38.331 timeToTrigger value.")
        if self.t_t310_ms not in {0, 50, 100, 200, 500, 1000, 2000}:
            raise ValueError("t_t310_ms is not a TS 38.331 T310 value.")
        if self.n310 not in {1, 2, 3, 4, 6, 8, 10, 20}:
            raise ValueError("n310 is not a TS 38.331 N310 value.")
        if self.n311 not in {1, 2, 3, 4, 5, 6, 8, 10}:
            raise ValueError("n311 is not a TS 38.331 N311 value.")
        self.t_ho_prep
        self.t_ho_exec
        self.t_mts
        self.t_rlfr
        self.t_t310
        self.a3_ttt

    def update(self, config_dict: dict):
        """
        Update the configuration with values from a dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config has no attribute '{key}'")
        self.validate()
