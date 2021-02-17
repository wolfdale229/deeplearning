'''Building a simple neural network'''


inputs =[100, 2, -3, -42, 15]
weights = [2, 0.6, 0.4, 0.3, -0.03]
bias = 0.098


def prediction(inputs : list, weight : list):
	"""This finds the weighted sum of inputs and weights"""

	total = 0.0 
	for i in range(0, len(inputs)):
		total += (inputs[i] * weights[i])
	if total >= 0:
		return 1
	else :
		return 0

print(prediction(inputs, weights))