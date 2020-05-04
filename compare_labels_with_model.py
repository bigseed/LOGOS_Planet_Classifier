import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.abs(x) * (x > 0)


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


class PlanetClassifier:
    def __init__(self, num_input, num_hidden1, num_hidden2, num_output):
        self.W_i = np.zeros((num_input, num_hidden1), dtype=np.float32)
        self.b_i = np.zeros((num_hidden1,), dtype=np.float32)

        self.W_h = np.zeros((num_hidden1, num_hidden2), dtype=np.float32)
        self.b_h = np.zeros((num_hidden2,), dtype=np.float32)

        self.W_h = np.zeros((num_hidden2, num_output), dtype=np.float32)
        self.b_h = np.zeros((num_output,), dtype=np.float32)

    def __call__(self, x):
        x = sigmoid(np.matmul(self.W_i, x) + self.b_i)
        x = sigmoid(np.matmul(self.W_h, x) + self.b_h)
        return softmax(np.matmul(self.W_o, x) + self.b_o)


dataset = np.load('January_dataset.npz')
inputs = dataset['inputs']
labels = dataset['labels']

model = PlanetClassifier(4, 120, 40, 9)

weights = np.load('January_parameters.npz')
model.W_i = weights['W_i']
model.b_i = weights['b_i']
model.W_h = weights['W_h']
model.b_h = weights['b_h']
model.W_o = weights['W_o']
model.b_o = weights['b_o']

outputs = list()
for pt, label in zip(inputs, labels):
    output = model(pt)
    outputs.append(np.argmax(output))
outputs = np.stack(outputs, axis=0)

colors = ('#c0392b', '#ecf0f1', '#f1c40f', '#f39c12',
          '#e74c3c', '#2ecc71', '#95a5a6', '#34495e', '#8e44ad')
tick_labels = ('Right_Ascension', 'Declination',
               'Rising Time', "Setting Time")

location = [231, 232, 233, 234, 235, 0, 236]
label_fig = plt.figure(figsize=(9, 5))
for i in range(3):
    for j in range(i + 1, 4):
        label_area = label_fig.add_subplot(location[2*i + j - 1])
        label_area.tick_params(labelbottom=False, labelleft=False,
                               bottom=False, left=False)
        label_area.set_xlabel(tick_labels[i])
        label_area.set_ylabel(tick_labels[j])
        for idx in range(9):
            mask = labels == idx
            plt.scatter(inputs[mask, i],
                        inputs[mask, j], c=colors[idx])

model_fig = plt.figure(figsize=(9, 5))
for i in range(3):
    for j in range(i + 1, 4):
        model_area = model_fig.add_subplot(location[2*i + j - 1])
        model_area.tick_params(labelbottom=False, labelleft=False,
                               bottom=False, left=False)
        model_area.set_xlabel(tick_labels[i])
        model_area.set_ylabel(tick_labels[j])
        for idx in range(9):
            mask = outputs == idx
            plt.scatter(inputs[mask, i],
                        inputs[mask, j], c=colors[idx])

plt.show()
label_fig.savefig('labels.png')
model_fig.savefig('models.png')
