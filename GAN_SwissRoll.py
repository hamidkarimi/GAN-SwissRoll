import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import random
import math

# This functions returns as many as numSamples samples from a multivarirant guassian distribution
def getMutiVarGaussian (numSamples, mean = [10, -10] , stdev =  [[10, -5], [-5, 10]]):	
	return np.random.multivariate_normal(mean, stdev, numSamples)

# Generating 3D Swiss Roll data
def generate3DSwissRoll(numSamples, mean = [10, -10] , stdev =  [[10, -5], [-5, 10]]):
	gaussianSamples =  getMutiVarGaussian(numSamples, mean, stdev)
	SRSamples = np.zeros([numSamples, 3 ])
	for i in range(numSamples):
		SRSamples[i][0] = gaussianSamples[i][0] * math.cos(gaussianSamples[i][0]) # x
		SRSamples[i][1] = gaussianSamples[i][1]					  # y
		SRSamples[i][2] = gaussianSamples[i][0] * math.sin(gaussianSamples[i][0]) # z

	return SRSamples

# returning next batch
def NextBach(Samples, batchSize):
	sampleSize = int(Samples.shape[0])
	for i in range(int(sampleSize/batchSize)):
		batch = Samples[i*batchSize:(i+1)*batchSize ] 
		yield batch

def getRandomBatch(Samples, batchSize):
	sampleSize = int(Samples.shape[0])
	d = int(sampleSize/batchSize)
	rnd = random.randint(0, d-1) 
	return Samples[rnd*batchSize:(rnd+1)*batchSize ] 

#Plotting generated and original data
def plotSwissRoll(original, generated = None, epoch = 1, title = 'Generative Adversial Network. Red: original Blue: generated. Epoch: ' ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	color = 'r'
	if generated != None :				
		ax.scatter(generated[:,0], generated[:,1], generated[:,2],  color='b', label="plot A")

	else:
		color = 'b'
	ax.scatter(original[:,0], original[:,1], original[:,2],  color=color)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.title(title + str(epoch))
	#fig.savefig("VAE-Epoch"+str(epoch)+".png")
	#plt.close(fig)
	#plt.show()
def plotPerformanceLog(log, percent = 0.1):
	step = int(log.shape[0] * percent)
	log = np.array(log)
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(log[:,0][::step],log[:,4][::step],'r') 
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Discriminator loss')

	fig1 = plt.figure()
	ax1 = fig1.gca()
	ax1.plot(log[:,0][::step],log[:,5][::step],'b') 
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Generator loss')

	#plt.show()

#Xavier initilaization for weights 
# From https://www.tensorflow.org/api_docs/python/contrib.layers/initializers#xavier_initializer

def weight_init(fan_in, fan_out, constant=1): 
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#====================================== Generative Adversarial Network ===============================

class GAN:
	def __init__(self, batchSize , zNum , discH1Size , discH2Size , genH1Size , genH2Size ,  learningRate  ):
		self.batchSize = batchSize # each batch is 100 consecutive points of (x,y,z) from generated swiss roll 
		self.zNum = zNum # number of z latent variables (  dimentionality of z distribution) 
		self.discH1Size = discH1Size # Number of hidden units in 1st layer of enconding multilayer perceptron 
		self.discH2Size = discH2Size # Number of hidden units in 2nd layer of enconding MLP 
		self.genH1Size = genH1Size # Number of hidden units in 1st layer of decoder MLP
		self.genH2Size = genH2Size # Number of hidden units in 2nd layer of decoder MLP  
		self.inputSize = 3 # each sample is a triple (x,y,z)
		self.learningRate = learningRate 
		self.input = tf.placeholder(tf.float32, [self.batchSize, self.inputSize]) # placeholder for input samples
		self.zs = tf.placeholder(tf.float32, [self.batchSize, self.zNum]) # placeholder for z noise samples

		self.initWeights()
		self.network()
		self.optimization()

		init = tf.initialize_all_variables()
		self.sess = tf.InteractiveSession()
		self.sess.run(init)

	def initWeights(self):
		self.Weights = dict()
		# Weights of discriminator network
		self.Weights['disc'] = {\
			'w1': tf.Variable(weight_init(self.inputSize, self.discH1Size),name="discw1"),\
			'w2': tf.Variable(weight_init(self.discH1Size, self.discH2Size),name="discw2"),\
			'w3': tf.Variable(weight_init(self.discH2Size, 1),name="discw3"),\
			'b1': tf.Variable(tf.zeros([self.discH1Size], dtype=tf.float32),name="discb1"),\
			'b2': tf.Variable(tf.zeros([self.discH2Size], dtype=tf.float32),name="discb2"),\
			'b3': tf.Variable(tf.zeros([1], dtype=tf.float32),name="discb3")}

		# Weights of generator network
		self.Weights['gen'] = {\
			'w1': tf.Variable(weight_init(self.zNum, self.genH1Size),name="genw1"),\
			'w2': tf.Variable(weight_init(self.genH1Size, self.genH2Size),name="genw1"),\
			'w3': tf.Variable(weight_init(self.genH2Size, self.inputSize),name="genw3"),\
			'b1': tf.Variable(tf.zeros([self.genH1Size], dtype=tf.float32),name="genb1"),\
			'b2': tf.Variable(tf.zeros([self.genH2Size], dtype=tf.float32),name="genb2"),\
			'b3': tf.Variable(tf.zeros([self.inputSize], dtype=tf.float32),name="genb3")}
	def network(self):
		self.generatedSample = self.generator()
		self.discRealProb, self.discRealLogit = self.discriminator(self.input) # Calling discriminator for actual data samples
		self.discGenProb, self.discGenLogit = self.discriminator(self.generatedSample, reuse=True) # Calling discriminator for generated data samples

	def optimization(self):
		self.discLossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discRealLogit, tf.ones_like(self.discRealLogit))) # logP(x)
		self.discLossGen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discGenLogit, tf.zeros_like(self.discGenLogit))) # 1-logP(z)
		self.discTotalloss = self.discLossReal + self.discLossGen

		self.genLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discGenLogit, tf.ones_like(self.discGenLogit))) #  logP(z)

		#------------------- Alternative statements -------------------
		"""self.discLossReal = tf.reduce_mean(tf.log(self.discRealProb))
		self.discLossGen = tf.reduce_mean(tf.log(1-self.discGenProb))
		self.discTotalloss = self.discLossReal + self.discLossGen
 
		self.genLoss =  tf.reduce_mean(tf.log(self.discGenProb))"""
		#--------------------------------------------------------------

		allVars = tf.trainable_variables()
		discVars = [var for var in allVars if 'disc' in var.name] # getting discriminator variables
		genVars = [var for var in allVars if 'gen' in var.name] # getting generator variables 

		self.discOptimal = tf.train.AdamOptimizer(self.learningRate).minimize(self.discTotalloss, var_list=discVars)
		self.genOptimal = tf.train.AdamOptimizer(self.learningRate).minimize(self.genLoss, var_list=genVars)

	def generator(self):
		l1 = tf.nn.tanh(tf.add(tf.matmul(self.zs, self.Weights['gen']['w1']), self.Weights['gen']['b1']))  # layer 1 tanh(z*W1+b1)
		l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.Weights['gen']['w2']), self.Weights['gen']['b2']))  # layer 2 tanh(l1*W2+b2)
		G = tf.add(tf.matmul(l2, self.Weights['gen']['w3']),self.Weights['gen']['b3']) # last layer is constructed sample
		return G

	def discriminator(self, Input, reuse = False ):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		l1 = tf.nn.tanh(tf.add(tf.matmul(Input, self.Weights['disc']['w1']), self.Weights['disc']['b1']))  # layer 1 tanh(X*W1+b1)
		l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.Weights['disc']['w2']), self.Weights['disc']['b2']))  # layer 2 tanh(l1*W2+b2)
		l3 = tf.nn.tanh(tf.add(tf.matmul(l2,self.Weights['disc']['w3']),self.Weights['disc']['b3'])) # layer 3 tanh(l2*W3+b3)
		return tf.nn.sigmoid(l3), l3 

	# balancing discriminator and generator, so they both keep up in the competition
	def balance(self,genPerformance, discPerformance, alternateThshld):
		if discPerformance - genPerformance > alternateThshld: # generator has left behind, so it is generator's turn on next iteration
			genTurn  = True
			discTurn = False

		elif genPerformance - discPerformance  > alternateThshld: # discriminator has left behind, so it is discriminator's turn on next iteration
			genTurn  = False
			discTurn = True

		elif abs(discPerformance - genPerformance) < alternateThshld: # performance of both are close to each other, so update both
			genTurn  = True
			discTurn = True

		return genTurn, discTurn

	def train(self,InputSamples, numEpoch, alternateThshld = 0.01 ):

		counter = 0
		display_z = np.random.uniform(-1, 1, [self.batchSize, self.zNum]).astype(np.float32)
		
		reportInterval = int((InputSamples.shape[0]/self.batchSize) * 0.1) # 10% of iterations of each epoch are reported and logged

		genTurn = discTurn = True # genTurn indicates generator's turn, discTurn is for discriminator's turn
		prevDerror = prevGerror = 10.0
		opt = lossfake =  lossreal = gloss = 0
		performanceLog = []
		for epoch in range(numEpoch):
			generatedXs = []
			for idx, batch in enumerate(NextBach(InputSamples, self.batchSize)):
				for k in xrange(5): # k is 5
					if discTurn:
						batch_z2 = np.random.uniform(-1, 1, [self.batchSize, self.zNum]).astype(np.float32)
						batch2= getRandomBatch(InputSamples,self.batchSize)  
						opt, lossfake, lossreal = self.sess.run([self.discOptimal,self.discLossGen, self.discLossReal ],feed_dict={ self.input: batch2, self.zs: batch_z2 })
				batch_z = np.random.uniform(-1, 1, [self.batchSize, self.zNum]).astype(np.float32)

				for k in xrange(1):
					if genTurn:
						opt, gloss = self.sess.run([self.genOptimal, self.genLoss ],feed_dict={ self.zs: batch_z })
	
				lossfake = self.discLossGen.eval({self.zs:  batch_z})
				lossreal = self.discLossReal.eval({self.input: batch})
				gloss = self.genLoss.eval({self.zs: batch_z})
				genPerformance =  (prevGerror - gloss) / prevGerror 
				discPerformance =  (prevDerror - (lossfake + lossreal)) / prevDerror
				################## balance ################
				genTurn , discTurn = self.balance(genPerformance, discPerformance, alternateThshld)
				###########################################
				prevDerror = lossfake + lossreal 
				prevGerror = gloss

				counter += 1					
				if idx % reportInterval == 0:
					discLossGen = self.discLossGen.eval({self.zs:  batch_z})
					discLossReal = self.discLossReal.eval({self.input: batch})
					genLoss = self.genLoss.eval({self.zs: batch_z})
					discGenProb = self.discGenProb.eval({self.zs:  batch_z})
					discRealProb = self.discRealProb.eval({self.input: batch})
					performanceLog.append((counter, epoch,discLossReal, discLossGen,discLossReal + discLossGen,genLoss,\
								 np.mean(discRealProb), np.mean(discGenProb), genTurn, discTurn ))
					print "epoch", epoch,'/',numEpoch, idx, "LossDisc(x):" ,discLossReal, "LossDisc(z):", discLossGen,"Total disc Loss:",discLossReal + discLossGen,\
						" generator loss", genLoss, "P(x in real|x) :", np.mean(discRealProb), "P(z in real|z):",np.mean(discGenProb), "generator Turn: ",genTurn, "discrimitor turn", discTurn 
					
			if epoch % 200 ==0 or (epoch <= 100 and epoch % 50 ==0) or (epoch <= 12 and epoch % 4 ==0):

				for i in range(int(InputSamples.shape[0] / self.batchSize)):
					batch_z2 = np.random.uniform(-1, 1, [self.batchSize, self.zNum]).astype(np.float32)
					sdata = self.sess.run([self.generatedSample],feed_dict={ self.zs: batch_z2})
					generatedXs.append(sdata) # harvesting generated samples 
				generatedXs = np.array(generatedXs)
				generatedXs = np.reshape(generatedXs, (InputSamples.shape[0],3))
				plotSwissRoll(original = generatedXs, generated = None, epoch = epoch+1, title ="Generative Adversial Network. Generated plot. Epoch: ")
				plotSwissRoll(InputSamples, generatedXs, epoch+1)
		return performanceLog

def main():
	InputSamples = generate3DSwissRoll(numSamples = 10000)
	gan = GAN(batchSize = 100, zNum = 2, discH1Size =200 , discH2Size = 200 , genH1Size = 200, genH2Size = 200,  learningRate = 0.001)
	performanceLog = gan.train(InputSamples, 401)
	performanceLog = np.array(performanceLog)
	with open("GANResults/logGAN2",'w') as  outputfile:
		for l in performanceLog:
			for k in range(len(l)):
				if k == len(l)-1:
					outputfile.write(str(l[k])+'\n')
				else:
					outputfile.write(str(l[k])+',')	

	plotPerformanceLog(performanceLog)
	#plt.show()

main()
	
