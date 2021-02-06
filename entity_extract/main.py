from data_prepare import DataPrepare
from bilsmt_crf import BiLSTM_CRF
from torch import optim
import torch
import os


def train(dp: DataPrepare):
    x_sent, x_tags, mask = dp.prepare_data()
    model = BiLSTM_CRF(len(dp.vocab), len(dp.tags))
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(300):  # normally you would NOT do 300 epochs, it is toy data

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = x_sent
        targets = x_tags

        # Step 3. Run our forward pass.
        loss = model.loss(sentence_in, targets, mask=mask)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    return model


def better_print(raw, path, dict):
    for sen, p in zip(raw, path):
        labels = {}
        p = [dict[x] for x in p]

        index = 0
        key = []
        value = ""
        while index < len(p):
            if 'B' in p[index] or 'I' in p[index]:
                key.append(sen[index])
                value = p[index]
            else:
                if (len(key) > 0):
                    keys = ''.join(key)
                    labels[keys] = value.split('_')[0]
                    key = []
                    value = ""
            index += 1

        print("语句是: %s" % sen)
        print(f"提取的实体有: {labels}")


if __name__ == '__main__':
    dp = DataPrepare(os.getcwd() + '/doccano.json')
    model = train(dp)
    with torch.no_grad():
        xx_sent, mask = dp.predict(["奥雷连诺上校在马孔多"])
        _, seqs = model(xx_sent, mask=mask)
        better_print(["奥雷连诺上校在马孔多"], seqs, dp.dict)
