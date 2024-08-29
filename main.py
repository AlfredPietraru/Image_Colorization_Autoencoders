import cv2
import torch
import preprocessing as prep
from net import NeuralNetwork
EPOCHS = 5
NR_IMAGES = 1401

def training_loop(model, loss_function, optimizer):
    model.train()
    for j in range(0, EPOCHS, 1):
        for i in range(1, NR_IMAGES, 1):
            L, ground_truth_values  = prep.return_lab_image(1)
            output = torch.Tensor(model(L)).squeeze(0)
            loss = loss_function(output, ground_truth_values)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("one epoch is over\n")
        torch.save(model.state_dict(), "./model.pth")

def evaluation(model, idx : int):
    model.eval()
    gray_img = prep.return_gray_image(idx)
    output : torch.Tensor = model(
        torch.Tensor(gray_img))
    output = output.to(torch.uint8)
    output = output.squeeze(0).squeeze(0)
    gray_img = gray_img.squeeze(0).squeeze(0)
    result = torch.stack([output[0], output[1], gray_img], dim=0)
    result = result.permute((1, 2, 0)).to(torch.uint8)
    cv2.imshow("gata", result.detach().numpy())
    cv2.waitKey(0)


model = NeuralNetwork()
# model.load_state_dict(torch.load("./model.pth"))
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
# training_loop(model, loss_function, optimizer)

model.load_state_dict(torch.load("./model.pth"))
evaluation(model, 468)