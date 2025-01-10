from torch.utils.tensorboard import SummaryWriter
import numpy as np

def example_add_histogram():
    writer = SummaryWriter()
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution', x + i, i)
        
    writer.close()


def example_scalar():
    writer = SummaryWriter()
    functions = {"sin": np.sin, "cos": np.cos, "tan": np.tan}

    for angle in range(-360, 360):
        radian = angle * np.pi/180
        for name, fun in functions.items():
            value = fun(radian)
            writer.add_scalar(name, value, angle)

    writer.close()

if __name__ == "__main__":  
    example_add_histogram()
    example_scalar()
"""
1. pip install tensorboard
2. run `tensorboard --logdir=runs`  
"""