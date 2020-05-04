import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class PlanetClassifier(tf.keras.Model):
    def __init__(self):
        super(PlanetClassifier, self).__init__()
        self.sequence = list()

        self.d1 = tf.keras.layers.Dense(120, activation='relu')
        self.d2 = tf.keras.layers.Dense(40, activation='relu')
        self.classifier = tf.keras.layers.Dense(9, activation='softmax')
        self.sequence.append(self.d1)
        self.sequence.append(self.d2)
        self.sequence.append(self.classifier)

    def call(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x


@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)


def deg2rad(deg):
    return deg * np.pi / 180


def calc_right_ascension(RA):
    RA = RA.split('_')

    hour = int(RA[0])
    minute = int(RA[1])
    second = int(RA[2])

    deg = hour * 15 + minute / 4 + second / 240
    return 100*np.sin(deg2rad(deg))


def calc_declination(dec):
    isNegative = False
    if not dec.find('-'):
        dec = dec[1:]
        isNegative = True

    dec = dec.split('_')

    hour = int(dec[0])
    minute = int(dec[1])
    second = int(dec[2])

    deg = hour + minute / 60 + second / 3600
    if isNegative:
        return -1 * 100*np.sin(deg2rad(deg))
    else:
        return 100*np.sin(deg2rad(deg))


def calc_time(time):
    time = time.split('_')
    hour = int(time[0])
    minute = int(time[1])

    return hour * 60.0 + minute


np.random.seed(19312)
num_planets = 9
sun = np.zeros((7, 4))
moon = np.zeros((7, 4))
mercury = np.zeros((7, 4))
venus = np.zeros((7, 4))
mars = np.zeros((7, 4))
jupiter = np.zeros((7, 4))
saturn = np.zeros((7, 4))
uranus = np.zeros((7, 4))
neptune = np.zeros((7, 4))
planets = np.array([sun, moon, mercury, venus, mars,
                    jupiter, saturn, uranus, neptune])
planet_index = {
    0: '태양',
    1: '달',
    2: '수성',
    3: '금성',
    4: '화성',
    5: '목성',
    6: '토성',
    7: '천왕성',
    8: '해왕성',
}
for idx in range(num_planets):
    with open('./csv/{}.csv'.format(planet_index[idx]), 'r') as f:
        coord = f.readlines()
        coord = list(map(lambda x: x.strip().split(','), coord))

        for i in range(1, len(coord)):
            planets[idx][i - 1][0] = calc_right_ascension(coord[i][1])
            planets[idx][i - 1][1] = calc_declination(coord[i][2])
            planets[idx][i - 1][2] = calc_time(coord[i][3])
            planets[idx][i - 1][3] = calc_time(coord[i][4])

pts, labels = list(), list()
for label, planet in enumerate(planets):
    for coord in planet[:4]:
        for _ in range(25):
            pts.append(coord + np.random.randn(*coord.shape))
            labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices(
    (pts, labels)).shuffle(len(pts)).batch(40)

model = PlanetClassifier()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

EPOCHS = 500
loss_per_epoch = list()
accuracy_per_epoch = list()


for epoch in range(EPOCHS):
    for x, label in train_ds:
        train_step(model, x, label, loss_object,
                   optimizer, train_loss, train_accuracy)

    template = 'Epoch {}\tLoss: {}\tAccuracy: {}%'
    loss = train_loss.result()
    accuracy = train_accuracy.result() * 100
    print(template.format(epoch + 1, loss, accuracy))

    loss_per_epoch.append(loss)
    accuracy_per_epoch.append(accuracy)

    train_loss.reset_states()
    train_accuracy.reset_states()


save_data = False
if save_data:
    np.savez_compressed('January_dataset.npz', inputs=pts, labels=labels)
    W_i, b_i = model.d1.get_weights()
    W_h, b_h = model.d2.get_weights()
    W_o, b_o = model.classifier.get_weights()
    W_i = np.transpose(W_i)
    W_h = np.transpose(W_h)
    W_o = np.transpose(W_o)
    np.savez_compressed('January_parameters.npz',
                        W_i=W_i, b_i=b_i,
                        W_h=W_h, b_h=b_h,
                        W_o=W_o, b_o=b_o)

fig = plt.figure(figsize=(11, 5))
loss_area = fig.add_subplot(121)
loss_area.plot(range(EPOCHS), loss_per_epoch)
loss_area.set_xlabel('EPOCHS')
loss_area.set_ylabel('Loss Value')
loss_area.set_title('Loss Value per EPOCH')

accuracy_area = fig.add_subplot(122)
accuracy_area.plot(range(EPOCHS), accuracy_per_epoch)
accuracy_area.set_xlabel('Accuracy(%)')
accuracy_area.set_title('Accuracy per EPOCH')

plt.show()
plt.savefig('result.png')
