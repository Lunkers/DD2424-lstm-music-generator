from music21 import converter, instrument, note, chord
import json




def load_data():
    file_arr = []
    dir_prefix = './maestro-v2.0.0/'
    notes = []
    with open('./maestro-v2.0.0/maestro-v2.0.0.json') as f:
        json_data = json.load(f)
        for row in json_data:
            if row['split'] == 'train':
                file_arr.append(row['midi_filename'])
        
   
        for file_name in file_arr[30:40]:
            midi = converter.parse(dir_prefix + file_name)
            print("Parsing %s", file_name)
            with open("notes.txt", 'a') as notefile:
                for element in midi.flat.notes:
                    if isinstance(element, note.Note):
                        notefile.write(str(element.pitch))
                    if isinstance(element, chord.Chord):
                        notefile.write('.'.join(str(n) for n in element.normalOrder))
        #print(len(set(notes)))
        #A5.C5.D5 ->[0 0 0 .. 1 0 ...] -> A5.C5.D5 -> chord.Chord(letter.split("."))
        # x[i] -> y[i+1] ≠ x[i+1]
        # D -> x = D[i:i+seq_length], y = D[i +1: i + seq_length]
        #A5.C5.D5 ->[0 0 0 .. 1 0 .. 1], [0 0 0 .. 0 1 .. 1] -> A5.C5.D5 ->
        #0.3.7 -> färre dimensioner, kanske inte lika träffsäkert
load_data()