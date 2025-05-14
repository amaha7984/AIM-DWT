import torch
import torch.nn as nn
import pywt
import ptwt

class HaarTransform(nn.Module):
    def __init__(self, level=1, mode="symmetric", with_grad=False):
        super().__init__()
        self.wavelet = pywt.Wavelet("haar")
        self.level = level
        self.mode = mode
        self.with_grad = with_grad

    def dwt(self, x):
        with torch.set_grad_enabled(self.with_grad):
            Yl, *Yh = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=self.level, mode=self.mode)
            if len(Yh) < 1 or len(Yh[0]) != 3:
                raise ValueError("Incorrect number of subbands after DWT.")
            xH, xV, xD = Yh[0]
            return Yl, xH, xV, xD

    def idwt(self, Yl, xH, xV, xD):
        with torch.set_grad_enabled(self.with_grad):
            return ptwt.waverec2([Yl.float(), (xH.float(), xV.float(), xD.float())], wavelet=self.wavelet)

    def forward(self, x, inverse=False):
        return self.idwt(*x) if inverse else self.dwt(x)
