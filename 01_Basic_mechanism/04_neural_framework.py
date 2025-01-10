import torch
import torch.nn as nn

class customModel(nn.Module):
    def __init__(self, inputs, classes, dropout=0.3):
        super(customModel, self).__init__()
        self.pipeline = nn.Sequential(
            # layer 1
            nn.Linear(inputs, 5),
            nn.ReLU(),
            # layer 2
            nn.Linear(5, 20),
            nn.ReLU(),
            # layer 3
            nn.Linear(20, classes),
            nn.Dropout(dropout),
            # output layer
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipeline(x)
    
    
if __name__ == "__main__":

    model = customModel(inputs=2, classes=3)
    print(model)

    v = torch.FloatTensor([[2, 3],])
    output = model(v)

    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():  
        print("cuda data: ", output.to('cuda'))


    """ for training loop we use this method """
    # for batch_x, batch_y in iterate_batches(data, batch_size=32):
        
    #     batch_x_t = torch.tensor(batch_x)
    #     batch_y_t = torch.tensor(batch_y)

    #     output_t = model(batch_x_t)
        
    #     loss_t = loss_function(output_t, batch_y_t)
    #     loss_t.backward()   

    #     optimizer.step()
    #     optimizer.zero_grad()
