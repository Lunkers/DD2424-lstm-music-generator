from model import LSTM
import torch.nn as nn
import torch
import torch.optim as optim
import json
from processing import one_hot_encoding, decode_output, loade_data



def main():
    notes = loade_data('./notes.json')['notes']
    int_to_sign =  loade_data('./int2sign.json')
    sign_to_int = loade_data('./sign2int.json')

    seq_length = 25

    
    #refactor this, we only need a one-hot for the input
    #select a sequence or whatever here, use predefined for now (testing)
    in_seq = convert_to_one_hot_matrix(notes[0:10], sign_to_int)
    ground_truth = target_tensor(notes[1:11], sign_to_int)
    print(in_seq.size())
    learning_rate = 0.001

    network = LSTM(hidden_size = 64, input_size = 90, output_size = 90)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), learning_rate)


    output, loss = train(network, criterion, in_seq, ground_truth, optimizer)
    print("output = {0}".format(output))
    print("loss = {0}".format(loss))

    #torch.save(network.state_dict(), PATH)
    

def train(network: LSTM, criterion, input_seq, follow_seq, optimizer: optim.Optimizer):
    follow_seq.unsqueeze_(-1)
    hidden = network.initHidden()
    memory = network.initMemory()
    loss = 0
    
    
    network.zero_grad()
    for i in range(input_seq.size()[0]):
        output, hidden, memory = network(input_seq[i], hidden, memory)
        l =  criterion(output, follow_seq[i])
        loss += l
    
        
    loss.backward()

    optimizer.step()

    return output, loss.item()

def convert_to_one_hot_matrix(data, sign_to_int):
    tensor = torch.zeros(len(data), 1, len(sign_to_int))
    for idx, note in enumerate(data):
        tensor[idx][0][sign_to_int[note]] = 1
    return tensor

def target_tensor(target_arr: list, sign_to_int: dict):
    """
    convert a sequence to a target tensor
    Args:
        target_arr: a list containing the targets for the input sequence, in string form
        sign_to_in: dictionary containing the mapping sign -> integer for the alphabet

    Returns:
        a 1D tensor with correct class values
    """
    indexes = [sign_to_int[sign] for sign in target_arr]
    return torch.LongTensor(indexes)

if __name__ == "__main__":
    main()