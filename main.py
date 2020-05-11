from model import LSTM
import torch.nn as nn
import torch
import torch.optim as optim
import json
from processing import one_hot_encoding, decode_output, loade_data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random


def main():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    notes = loade_data('./notes.json')['notes']
    validation = loade_data('./validation.json')['notes']
    test = loade_data('./test.json')['notes']
    int_to_sign =  loade_data('./int2sign.json')
    sign_to_int = loade_data('./sign2int.json')
    seq_length = 25

    
    #refactor this, we only need a one-hot for the input
    #select a sequence or whatever here, use predefined for now (testing)
    in_seq = convert_to_one_hot_matrix(notes[0:10], sign_to_int)
    ground_truth = target_tensor(notes[1:11], sign_to_int)
    learning_rate = 0.001

    network = LSTM(hidden_size = 64, input_size = 90, output_size = 90)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), learning_rate)
    # move network to GPU
    network.to(device)
    print(device)
    network, _, losses = trainLoop(network, criterion, notes, optimizer, 1, 25, sign_to_int)
    plt.figure()
    plt.plot(losses)
    plt.show()
    acc = evaluateAccuracy(validation, network, 25, sign_to_int)
    print(acc)

    save_network(network, "net.pth")
    

def trainLoop(network: LSTM, criterion, data: list, optimizer: optim.Optimizer, n_epochs: int, seq_length:int, sign_to_int):
    iters_per_epoch = len(data) // seq_length
    iters = iters_per_epoch * n_epochs
    total_loss = 0
    all_loss = []
    print( "Training for %d iterations" % iters)
    for iteration in range(1, iters + 1):
        if iteration % 1000 == 0:
            print(iteration)
        input_seq, follow_seq = randomTrainingExample(data, seq_length, sign_to_int)
        #move data to correct device
        input_seq.to(device)
        follow_seq.to(device)
        output, loss = train(network, criterion, input_seq, follow_seq, optimizer)
        total_loss += loss
        all_loss.append(loss)

    return network, total_loss, all_loss

def evaluateAccuracy(data: list, network: LSTM, seq_length: int, sign_to_int):
    network.eval()
    hidden = network.initHidden()
    memory = network.initMemory()
    print(len(data))
    right = 0
    total = 0

    for i in range(0, len(data), seq_length):
        in_seq = convert_to_one_hot_matrix(data[i:i+ seq_length], sign_to_int)
        out_seq = target_tensor(data[i+1: i+ seq_length + 1], sign_to_int)
        in_seq.to(device)
        out_seq.to(device)
        out_seq.unsqueeze_(-1)
        for j in range(out_seq.size()[0]):
            output, hidden, memory = network(in_seq[j], hidden, memory)
            _, guess = output.max(1)
            if guess == out_seq[j]:
                right = right + 1
            total = total + 1

    return right / total
        


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
    """
    convert a sequence to a matrix of one-hot encoded vectors
    Args:
        data: the data sequence to encode
        sign_to_int: mapping from sign to integer value
    """
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


def synthesize_notes(network: LSTM, start_input, n: int, sign_to_int: dict, int_to_sign: dict):
    seq = []
    hidden = network.initHidden()
    memory = network.initMemory()
    
    with torch.no_grad():    
        for i in range(n): 
            p, hidden, memory = network(inputs, hidden, memory)

            p = p.numpy()[0]
            ind = np.random.choice((p.shape[0]), 1, p=p)[0]

            inputs = torch.zero(1, 1, len(sign_to_int))
            inputs = inputs[0][0][ind] = 1
            
            seq.append(ind_to_char[str(ind)])

    return seq

def randomTrainingExample(data, seq_length, sign_to_int):
    i = random.randint(0, len(data) - seq_length - 1)
    
    return  convert_to_one_hot_matrix(data[i:i+seq_length], sign_to_int), target_tensor(data[i+1: i + seq_length +1], sign_to_int)

def save_network(network: LSTM, path: str):
    torch.save(network.state_dict(), path)

def load_network(network: LSTM, path: str):
    network.load_state_dict(torch.load(path))

if __name__ == "__main__":
    main()