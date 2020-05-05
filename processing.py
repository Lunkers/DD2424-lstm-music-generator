from music21 import converter, instrument, note, chord
import json
import torch

nextString = "<NEXT>"
endString = "<END>"


def load_data():
    file_arr = []
    dir_prefix = './maestro-v2.0.0/'
    notes = []
    with open('./maestro-v2.0.0/maestro-v2.0.0.json') as f:
        json_data = json.load(f)
        for row in json_data:
            if row['split'] == 'train':
                file_arr.append(row['midi_filename'])
        
   
    #parse
    #lägg in alla noter i vektor, <NEXT> där det behövs
    #i slutet av filen , lägg till en <END>-token
    #när alla filer parsade, returnera notes
    notes = []

    for file_name in file_arr:
        midi = converter.parse(dir_prefix + file_name)
        print("Parsing {0}".format(file_name))
        for element in midi.flat.notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            if isinstance(element, chord.Chord):
                for n in element.pitches:
                    notes.append(str(n))
            notes.append(nextString)
        notes.append(endString)

    data = {"notes":notes}


    #spara notes som JSON-fil
    #JSON.dumps(s)
    with open("notes.json", 'a') as notefile:
        json.dump(data, notefile)

def read_data(file_path):
    data = JSON.load(file_path)
    data = data["notes"]
    alphabet = set(data)
    sign_to_int = dict([(y,x) for x,y in enumerate(sorted(alphabet))])
    int_to_sign = dict([(x,y) for x,y in enumerate(sorted(alphabet))])

    return sign_to_int, int_to_sign


def one_hot_encoding(sign_int: int, alphabet_size: int):

    """get one-hot encoded tensor for a sign in an alphabet
        Args:
            sign_int: sign in alphabet converted to integer value
            alphabet_size: total size of alphabet
        
        Returns:
            a one-hot encoded tensor of size [1 x alphabet_size]
    """

    one_hot = torch.zeros(1, alphabet_size)
    one_hot[0][sign_int] = 1
    return one_hot
    
def decode_output(prob_vec:torch.tensor, int_to_sign: dict):
    """
        Get a sign in our alphabet from a prboability vector

        Args:
            prob_vec: a tensor of size [1 x alphabet size], containing probabilities for each sign
            int_to_sign: a dictionary containing mappings from indices (ints) to signs 

        Returns:
            The sign with the highest probability

    """
    _ , index = prob_vec.max(1)
    return int_to_sign[index[0].item()]
    
def build_midi():
    pass




load_data()
sign2int, int2sign = read_data('./notes.json')
with open("sign2int.json") as f:
    json.dump(sign2int, f)

with open("int2sign.json") as f:
    json.dump(int2sign, f)