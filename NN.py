def neural_network(inputs: list, weights: list) -> float:   
    prediction = 0
    for i in range(len(inputs)):
        prediction += inputs[i] * weights[i]
    return prediction

def ele_mul(delta: float, training_vector: list) -> list:
    weights_deltas = []
    for i in range(len(training_vector)):
        weights_deltas.append(delta * training_vector[i])
    return weights_deltas

toes = [8.5, 9.5, 9.9, 9.0]
win_lose_percentage = [0.65, 0.8, 0.8, 0.9]
n_fans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]

input = [toes[0], win_lose_percentage[0], n_fans[0]]
weight = [0.1, 0.2, -0.1]
alpha = 0.1

for iteration in range(len(win_or_lose_binary)):
    prediction = neural_network(input, weight)
    error = (prediction - true) ** 2
    delta = prediction - true
    weight_deltas = ele_mul(delta, input)
    print("prediction: ", prediction)
    print("weights: ", weight)
    for i in range(len(weight)):
        weight[i] -= alpha * weight_deltas[i]
