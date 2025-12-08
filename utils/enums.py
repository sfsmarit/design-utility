from enum import Enum


class Unit(Enum):
    ABS = ("Abs", 1)
    DB = ("dB", 1)
    VSWR = ("VSWR", 1)
    SMITH = ("Smith", 1)
    MAXGAIN = ("MaxGain", 1)
    Y_DB = ("Y_dB", 1)
    Y_REAL = ("Y_real", 1)
    Y_IMAG = ("Y_imag", 1)
    GHZ = ("GHz", 1e9)
    MHZ = ("MHz", 1e6)
    KHZ = ("kHz", 1e3)
    HZ = ("Hz", 1)
    SEC = ("s", 1)
    MSEC = ("ms", 1e-3)
    USEC = ("us", 1e-6)
    NSEC = ("ns", 1e-9)
    PSEC = ("ps", 1e-12)

    def __str__(self) -> str:
        return self.value[0]

    @property
    def digits(self) -> float:
        return self.value[1]
