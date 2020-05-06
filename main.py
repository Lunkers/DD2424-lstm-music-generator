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

    
    one_hot_matrix = convert_to_one_hot_matrix(notes, sign_to_int)
    print(one_hot_matrix.size())
    learning_rate = 0.001

    network = LSTM(hidden_size = 64, input_size = 90, output_size = 90)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), learning_rate)

    output, loss = train(network, criterion, one_hot_matrix[0:10], one_hot_matrix[1:11], optimizer)
    print("output = {0}".format(output))
    print("loss = {0}".format(loss))

    #torch.save(network.state_dict(), PATH)
    

def train(network: LSTM, criterion, input_seq, follow_seq, optimizer: optim.Optimizer):
    hidden = network.initHidden()
    memory = network.initMemory()
    loss = 0
    print(input_seq.size())
    print(follow_seq.size())
    
    
    network.zero_grad()
    
    output, hidden, memory = network(input_seq, hidden, memory)
        
    loss = criterion(output, follow_seq.squeeze())
        
    loss.backward()

    optimizer.step()

    return output, loss.item()

def convert_to_one_hot_matrix(data, sign_to_int):
    tensor = torch.zeros(len(data), 1, len(sign_to_int))
    for idx, note in enumerate(data):
        tensor[idx][0][sign_to_int[note]] = 1
    return tensor

if __name__ == "__main__":
    main()