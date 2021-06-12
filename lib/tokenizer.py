import re

import numpy as np


class Tokenizer(object):

    def __init__(self):
        self.stoi = {
            '(': 0, ')': 1, '+': 2, ',': 3, '-': 4, '/b': 5, '/c': 6, '/h': 7, '/i': 8, '/m': 9, '/s': 10,
            '/t': 11, '0': 12, '1': 13, '10': 14, '100': 15, '101': 16, '102': 17, '103': 18, '104': 19, '105': 20,
            '106': 21, '107': 22, '108': 23, '109': 24, '11': 25, '110': 26, '111': 27, '112': 28, '113': 29, '114': 30,
            '115': 31, '116': 32, '117': 33, '118': 34, '119': 35, '12': 36, '120': 37, '121': 38, '122': 39, '123': 40,
            '124': 41, '125': 42, '126': 43, '127': 44, '128': 45, '129': 46, '13': 47, '130': 48, '131': 49, '132': 50,
            '133': 51, '134': 52, '135': 53, '136': 54, '137': 55, '138': 56, '139': 57, '14': 58, '140': 59, '141': 60,
            '142': 61, '143': 62, '144': 63, '145': 64, '146': 65, '147': 66, '148': 67, '149': 68, '15': 69, '150': 70,
            '151': 71, '152': 72, '153': 73, '154': 74, '155': 75, '156': 76, '157': 77, '158': 78, '159': 79, '16': 80,
            '161': 81, '163': 82, '165': 83, '167': 84, '17': 85, '18': 86, '19': 87, '2': 88, '20': 89, '21': 90,
            '22': 91, '23': 92, '24': 93, '25': 94, '26': 95, '27': 96, '28': 97, '29': 98, '3': 99, '30': 100,
            '31': 101, '32': 102, '33': 103, '34': 104, '35': 105, '36': 106, '37': 107, '38': 108, '39': 109, '4': 110,
            '40': 111, '41': 112, '42': 113, '43': 114, '44': 115, '45': 116, '46': 117, '47': 118, '48': 119, '49': 120,
            '5': 121, '50': 122, '51': 123, '52': 124, '53': 125, '54': 126, '55': 127, '56': 128, '57': 129, '58': 130,
            '59': 131, '6': 132, '60': 133, '61': 134, '62': 135, '63': 136, '64': 137, '65': 138, '66': 139, '67': 140,
            '68': 141, '69': 142, '7': 143, '70': 144, '71': 145, '72': 146, '73': 147, '74': 148, '75': 149, '76': 150,
            '77': 151, '78': 152, '79': 153, '8': 154, '80': 155, '81': 156, '82': 157, '83': 158, '84': 159, '85': 160,
            '86': 161, '87': 162, '88': 163, '89': 164, '9': 165, '90': 166, '91': 167, '92': 168, '93': 169, '94': 170,
            '95': 171, '96': 172, '97': 173, '98': 174, '99': 175, 'B': 176, 'Br': 177, 'C': 178, 'Cl': 179, 'D': 180,
            'F': 181, 'H': 182, 'I': 183, 'N': 184, 'O': 185, 'P': 186, 'S': 187, 'Si': 188, 'T': 189, '<sos>': 190,
            '<eos>': 191, '<pad>': 192}
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def split_form(self, form):
        string = ''
        for i in re.findall(r"[A-Z][^A-Z]*", form):
            elem = re.match(r"\D+", i).group()
            num = i.replace(elem, "")
            if num == "":
                string += f"{elem} "
            else:
                string += f"{elem} {str(num)} "
        return string.rstrip(' ')

    def split_form2(self, form):
        string = ''
        for i in re.findall(r"[a-z][^a-z]*", form):
            elem = i[0]
            num = i.replace(elem, "").replace('/', "")
            num_string = ''
            for j in re.findall(r"[0-9]+[^0-9]*", num):
                num_list = list(re.findall(r'\d+', j))
                assert len(num_list) == 1, f"len(num_list) != 1"
                _num = num_list[0]
                if j == _num:
                    num_string += f"{_num} "
                else:
                    extra = j.replace(_num, "")
                    num_string += f"{_num} {' '.join(list(extra))} "
            string += f"/{elem} {num_string}"
        return string.rstrip(' ')

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            i = (i + 192) % 193
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            elif i == self.stoi['<sos>']:
                continue
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

    def tokenize(self, inchi, max_len=277):
        inchi_1 = self.split_form(inchi.split('/')[1])
        inchi_others = self.split_form2('/'.join(inchi.split('/')[2:]))
        inchi = inchi_1 + ' ' + inchi_others
        try:
            x = np.array(self.text_to_sequence(inchi), np.int32)
            assert len(x) <= max_len, len(x)
            x = np.pad(x, (0, max_len - len(x)),
                       constant_values=self.stoi["<pad>"])
            return x
        except:
            return np.array([self.stoi['<pad>']] * max_len)

    def get_length(self, text):
        seq = self.text_to_sequence(text)
        length = len(seq) - 2
        return length
