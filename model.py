''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.
Example:
 
	model = NeuralNetwork()
	sample = [input_1, input_2, input_3, input_4, ...]
	outputs = model.calculate_outputs(sample)
 
 
Inputs Names: 
	0) cal_baja
	1) cal_media
	2) cal_alta
	3) prot_baja
	4) prot_media
	5) prot_alta
	6) grasa_baja
	7) grasa_media
	8) grasa_alta

You can predict with a batch of samples using calculate_batch_output method	 
IMPORTANT: input batch must be <class 'numpy.ndarray'> type
Example_1:
	model = NeuralNetwork()
	input_batch = np.array([[1, 2], [4, 5]])
	outputs = model.calculate_batch_output(input_batch)
Example_2:
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})
	outputs = model.calculate_batch_output(input_batch.values)
''' 
import numpy as np
import pandas as pd

class NeuralNetwork:

	def __init__(self):
		self.inputs_number = 9
		self.input_names = ['cal_baja', 'cal_media', 'cal_alta', 'prot_baja', 'prot_media', 'prot_alta', 'grasa_baja', 'grasa_media', 'grasa_alta']

	@staticmethod
	def Linear(x):
		return x

	@staticmethod
	def HyperbolicTangent(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		cal_baja = inputs[0]
		cal_media = inputs[1]
		cal_alta = inputs[2]
		prot_baja = inputs[3]
		prot_media = inputs[4]
		prot_alta = inputs[5]
		grasa_baja = inputs[6]
		grasa_media = inputs[7]
		grasa_alta = inputs[8]

		scaled_cal_baja = (cal_baja-0.2787550092)/0.2977240086
		scaled_cal_media = (cal_media-0.2574619949)/0.3272700012
		scaled_cal_alta = (cal_alta-0.07529579848)/0.1918690056
		scaled_prot_baja = (prot_baja-0.3887990117)/0.3655410111
		scaled_prot_media = (prot_media-0.1561399996)/0.2783190012
		scaled_prot_alta = (prot_alta-0.07082600147)/0.1998849958
		scaled_grasa_baja = (grasa_baja-0.1844000071)/0.2919149995
		scaled_grasa_media = (grasa_media-0.2757669985)/0.3267289996
		scaled_grasa_alta = (grasa_alta-0.09818200022)/0.2333019972
		dense2d_layer_1_output_0 = self.HyperbolicTangent( -0.0050711 + (-0.0659136*scaled_cal_baja) + (-0.00612734*scaled_cal_media) + (0.089019*scaled_cal_alta) + (-0.0332352*scaled_prot_baja) + (0.026356*scaled_prot_media) + (-0.06074*scaled_prot_alta) + (0.123094*scaled_grasa_baja) + (-0.0141973*scaled_grasa_media) + (-0.0971432*scaled_grasa_alta) )
		dense2d_layer_1_output_1 = self.HyperbolicTangent( -0.00927948 + (-0.0572974*scaled_cal_baja) + (-0.0116639*scaled_cal_media) + (0.0872322*scaled_cal_alta) + (-0.0267857*scaled_prot_baja) + (0.0119004*scaled_prot_media) + (-0.0583536*scaled_prot_alta) + (0.119409*scaled_grasa_baja) + (-0.0194337*scaled_grasa_media) + (-0.101126*scaled_grasa_alta) )
		dense2d_layer_1_output_2 = self.HyperbolicTangent( -0.61518 + (0.0474323*scaled_cal_baja) + (0.465999*scaled_cal_media) + (-0.203004*scaled_cal_alta) + (0.478336*scaled_prot_baja) + (-0.197135*scaled_prot_media) + (0.261621*scaled_prot_alta) + (-0.087562*scaled_grasa_baja) + (-0.164341*scaled_grasa_media) + (0.575551*scaled_grasa_alta) )
		dense2d_layer_1_output_3 = self.HyperbolicTangent( 0.00364586 + (0.0680575*scaled_cal_baja) + (0.00457016*scaled_cal_media) + (-0.0895997*scaled_cal_alta) + (0.0349061*scaled_prot_baja) + (-0.0306723*scaled_prot_media) + (0.0614194*scaled_prot_alta) + (-0.123718*scaled_grasa_baja) + (0.0128628*scaled_grasa_media) + (0.0957981*scaled_grasa_alta) )
		dense2d_layer_1_output_4 = self.HyperbolicTangent( 0.00330117 + (0.0698249*scaled_cal_baja) + (0.00328752*scaled_cal_media) + (-0.090121*scaled_cal_alta) + (0.0360638*scaled_prot_baja) + (-0.0327681*scaled_prot_media) + (0.0618299*scaled_prot_alta) + (-0.124871*scaled_grasa_baja) + (0.0121219*scaled_grasa_media) + (0.0959971*scaled_grasa_alta) )
		dense2d_layer_1_output_5 = self.HyperbolicTangent( 0.00361052 + (0.069298*scaled_cal_baja) + (0.00368482*scaled_cal_media) + (-0.0899646*scaled_cal_alta) + (0.0356661*scaled_prot_baja) + (-0.0317753*scaled_prot_media) + (0.0616685*scaled_prot_alta) + (-0.124686*scaled_grasa_baja) + (0.0124213*scaled_grasa_media) + (0.0962441*scaled_grasa_alta) )
		dense2d_layer_1_output_6 = self.HyperbolicTangent( -0.00811389 + (-0.0593491*scaled_cal_baja) + (-0.0104584*scaled_cal_media) + (0.0875413*scaled_cal_alta) + (-0.0283868*scaled_prot_baja) + (0.015597*scaled_prot_media) + (-0.0589459*scaled_prot_alta) + (0.120157*scaled_grasa_baja) + (-0.0180307*scaled_grasa_media) + (-0.0997512*scaled_grasa_alta) )
		dense2d_layer_1_output_7 = self.HyperbolicTangent( 0.101118 + (0.267801*scaled_cal_baja) + (-0.0996068*scaled_cal_media) + (-0.275959*scaled_cal_alta) + (0.144767*scaled_prot_baja) + (-0.130663*scaled_prot_media) + (-0.269958*scaled_prot_alta) + (0.0449309*scaled_grasa_baja) + (0.267989*scaled_grasa_media) + (-0.445483*scaled_grasa_alta) )
		dense2d_layer_1_output_8 = self.HyperbolicTangent( 0.0115175 + (0.0541006*scaled_cal_baja) + (0.013315*scaled_cal_media) + (-0.0870187*scaled_cal_alta) + (0.0241327*scaled_prot_baja) + (-0.00547005*scaled_prot_media) + (0.0573401*scaled_prot_alta) + (-0.118583*scaled_grasa_baja) + (0.0219691*scaled_grasa_media) + (0.104325*scaled_grasa_alta) )
		dense2d_layer_1_output_9 = self.HyperbolicTangent( 0.144557 + (-0.129831*scaled_cal_baja) + (-0.25711*scaled_cal_media) + (0.120626*scaled_cal_alta) + (-0.114375*scaled_prot_baja) + (0.200933*scaled_prot_media) + (0.299707*scaled_prot_alta) + (-0.567591*scaled_grasa_baja) + (-0.0536405*scaled_grasa_media) + (-0.0363608*scaled_grasa_alta) )
		dense2d_layer_2_output_0 = self.HyperbolicTangent( 0.0574729 + (0.130529*dense2d_layer_1_output_0) + (0.127987*dense2d_layer_1_output_1) + (0.42384*dense2d_layer_1_output_2) + (-0.13109*dense2d_layer_1_output_3) + (-0.132193*dense2d_layer_1_output_4) + (-0.132006*dense2d_layer_1_output_5) + (0.128323*dense2d_layer_1_output_6) + (0.292191*dense2d_layer_1_output_7) + (-0.128097*dense2d_layer_1_output_8) + (0.360838*dense2d_layer_1_output_9) )
		dense2d_layer_2_output_1 = self.HyperbolicTangent( -0.0516318 + (-0.120111*dense2d_layer_1_output_0) + (-0.117531*dense2d_layer_1_output_1) + (-0.382479*dense2d_layer_1_output_2) + (0.120701*dense2d_layer_1_output_3) + (0.12176*dense2d_layer_1_output_4) + (0.121569*dense2d_layer_1_output_5) + (-0.117898*dense2d_layer_1_output_6) + (-0.269446*dense2d_layer_1_output_7) + (0.117539*dense2d_layer_1_output_8) + (-0.328885*dense2d_layer_1_output_9) )
		dense2d_layer_2_output_2 = self.HyperbolicTangent( 0.0473671 + (0.113567*dense2d_layer_1_output_0) + (0.111018*dense2d_layer_1_output_1) + (0.358143*dense2d_layer_1_output_2) + (-0.114159*dense2d_layer_1_output_3) + (-0.115181*dense2d_layer_1_output_4) + (-0.114993*dense2d_layer_1_output_5) + (0.111389*dense2d_layer_1_output_6) + (0.254882*dense2d_layer_1_output_7) + (-0.110987*dense2d_layer_1_output_8) + (0.308959*dense2d_layer_1_output_9) )
		approximation_layer_output_0 = self.Linear( 0.0480327 + (0.718321*dense2d_layer_2_output_0) + (-0.654169*dense2d_layer_2_output_1) + (0.614702*dense2d_layer_2_output_2) )
		unscaling_layer_output_0=approximation_layer_output_0*0.6535760164+4.104249954
		rating = max(0, unscaling_layer_output_0)
		outputs = [rating]
		return outputs

	def calculate_batch_output(self, input_batch):
		output_batch = np.zeros((len(input_batch), 1))
		for i in range(len(input_batch)):
			inputs = list(input_batch[i])
			output = self.calculate_outputs(inputs)
			output_batch[i] = output
		return output_batch

def main():

	# Introduce your input values here
	cal_baja = 0  # cal_baja
	cal_media = 0  # cal_media
	cal_alta = 0  # cal_alta
	prot_baja = 0  # prot_baja
	prot_media = 0  # prot_media
	prot_alta = 0  # prot_alta
	grasa_baja = 0  # grasa_baja
	grasa_media = 0  # grasa_media
	grasa_alta = 0  # grasa_alta

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(cal_baja)
	inputs.append(cal_media)
	inputs.append(cal_alta)
	inputs.append(prot_baja)
	inputs.append(prot_media)
	inputs.append(prot_alta)
	inputs.append(grasa_baja)
	inputs.append(grasa_media)
	inputs.append(grasa_alta)

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
