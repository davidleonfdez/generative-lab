from genlab.style_transfer import *


class TestContentLoss:
    def test():
        x = torch.Tensor([[[0]*4]*4]*3)
        y = torch.Tensor([[[1]*4]*4]*3)
        loss1 = content_loss(x, y)
        x += 0.5
        loss2 = content_loss(x, y)
        y -= 0.25
        loss3 = content_loss(x, y)
        assert loss1 == 4*4*3
        assert loss2 == 4*3
        assert loss3 == 3
