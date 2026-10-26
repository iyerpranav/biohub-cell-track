import torch, torch.nn as nn

def block(i, o):
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
        nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True)
        )

class UNet3D(nn.Module):
    # anisotropic pooling: z is only 16 slices, so don't squash it early
    def __init__(self, c=16):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(1, c), block(c, 2*c), block(2*c, 4*c), block(4*c, 8*c)
        self.p1, self.p2, self.p3 = nn.MaxPool3d((1,2,2)), nn.MaxPool3d(2), nn.MaxPool3d(2)
        self.u3 = nn.ConvTranspose3d(8*c, 4*c, 2, stride=2)
        self.u2 = nn.ConvTranspose3d(4*c, 2*c, 2, stride=2)
        self.u1 = nn.ConvTranspose3d(2*c, c, (1,2,2), stride=(1,2,2))
        self.d3, self.d2, self.d1 = block(8*c, 4*c), block(4*c, 2*c), block(2*c, c)
        self.out = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.b(self.p3(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))
