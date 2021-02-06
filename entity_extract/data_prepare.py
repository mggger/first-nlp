import json_lines
import torch


class DataPrepare:
    def __init__(self, file):
        self.file = file
        self.raw = []
        self.data = self._prepare_data()
        self.vocab, self.tags = self._prepare_vocab_tag()
        self.senlens = self._prepare_max_length()

        self.dict = {value: key for key, value in self.tags.items()}


    # step1, convert data to
    def _prepare_data(self):
        data = []
        with open(self.file, 'rb') as f:
            for item in json_lines.reader(f):
                self.raw.append(item['text'])
                txt = list(map(lambda x: x, item['text']))
                lables = item['labels']
                length = len(txt)

                tag = ['O' for i in range(0, length)]
                for label in lables:
                    start, end, name = label
                    tag[start] = f"{name}_B"
                    for i in range(start + 1, end):
                        tag[i] = f"{name}_I"

                data.append((txt, tag))

        return data

    def _prepare_vocab_tag(self):

        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2
        }

        tas = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2
        }

        for sentence, tags in self.data:
            for word, tag in zip(sentence, tags):
                if word not in vocab:
                    vocab[word] = len(vocab)
                if tag not in tas:
                    tas[tag] = len(tas)

        return vocab, tas

    def _prepare_max_length(self, maxsen=32):
        lens = [len(s[0]) for s in self.data]
        return max(max(lens), maxsen)

    def prepare_sentence(self, sentences):
        return torch.tensor([self.vocab[w] for w in sentences], dtype=torch.long)

    def predict(self, sens):

        def fill_function(sentence, maxlens):
            return [1 for i in range(len(sentence))] + [0 for i in range(maxlens - len(sentence))]

        x_sent = torch.full((len(sens), self.senlens), 0, dtype=torch.long)
        for index, sn in enumerate(sens):
            tsn = self.prepare_sentence(sn)
            x_sent[index, : tsn.shape[0]] = tsn

        mask = torch.full((len(sens), self.senlens), 0, dtype=torch.long)
        for index, sn in enumerate(sens):
            fills = fill_function(sn, self.senlens)
            fillstensro = torch.tensor(fills, dtype=torch.long)
            mask[index: fillstensro.shape[0]] = fillstensro
        return x_sent, mask

    def prepare_tags(self, sentences):
        return torch.tensor([self.tags[w] for w in sentences], dtype=torch.long)

    def ids_to_tags(self, sens):
        return [self.tags[x] for x in sens]

    def prepare_data(self):
        sens = [x[0] for x in self.data]
        tags = [x[1] for x in self.data]
        x_sent = torch.full((len(sens), self.senlens), 0, dtype=torch.long)
        x_tags = torch.full((len(tags), self.senlens), 0, dtype=torch.long)
        for index, sn in enumerate(sens):
            tsn = self.prepare_sentence(sn)
            x_sent[index, : tsn.shape[0]] = tsn

        for index, tag in enumerate(tags):
            ttag = self.prepare_tags(tag)
            x_tags[index, : ttag.shape[0]] = ttag

        mask = (x_tags != 0).float()

        return x_sent, x_tags, mask

    def revert(self, sens):

        return [self.dict[x] for x in sens]

