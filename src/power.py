"""
Local LLM energy and emissions tracking via CodeCarbon.

Usage
-----
    from src.power import measure_power

    with measure_power() as m:
        run_inference(...)

    print(m.joules)       # energy in joules
    print(m.emissions_g)  # CO2 in grams (CodeCarbon estimate)
"""

from __future__ import annotations


class CodeCarbonMeter:
    """Context manager using CodeCarbon to track energy and CO2 for local inference."""

    def __init__(self):
        from codecarbon import EmissionsTracker  # type: ignore
        # log_level (not logging_level) is the correct kwarg
        self.tracker = EmissionsTracker(save_to_file=False, log_level="critical")
        self.joules: float = 0.0
        self.emissions_g: float = 0.0
        self.method: str = "CodeCarbon"

    def __enter__(self) -> "CodeCarbonMeter":
        self.tracker.start()
        return self

    def __exit__(self, *_) -> None:
        self.tracker.stop()
        # final_emissions_data is populated after stop()
        data = self.tracker.final_emissions_data
        kwh = data.energy_consumed          # kWh
        self.joules = kwh * 3_600_000       # kWh → J
        self.emissions_g = data.emissions * 1000  # kg CO2 → g

    def kwh(self) -> float:
        return self.joules / 3_600_000


def measure_power() -> CodeCarbonMeter:
    return CodeCarbonMeter()


def check_power_available() -> tuple[bool, str]:
    """CodeCarbon works on any platform — always available."""
    return True, "CodeCarbon (cross-platform estimate)"
