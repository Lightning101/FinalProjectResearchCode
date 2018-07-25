/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neatprototyperesearch;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.simple.EncogUtility;

/**
 *
 * @author sean
 */
public class NEATPrototypeResearch {

    /**
     * @param args the command line arguments
     */

       public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
			{ 0.0, 1.0 }, { 1.0, 1.0 } };

	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

	public static void main(final String args[]) {

		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
		NEATPopulation pop = new NEATPopulation(2,1,1000);
		pop.setInitialConnectionDensity(1.0);
                pop.setNEATActivationFunction( new ActivationReLU());
		pop.reset();

		CalculateScore score = new TrainingSetScore(trainingSet);
		// train the neural network
		
		final EvolutionaryAlgorithm train = NEATUtil.constructNEATTrainer(pop,score);
		
		do {
			train.iteration();
			System.out.println("Epoch #" + train.getIteration() + " Error:" + train.getError()+ ", Species:" + pop.getSpecies().size());
		} while(train.getError() > 0.01);

		NEATNetwork network = (NEATNetwork)train.getCODEC().decode(train.getBestGenome());

		// test the neural network
		System.out.println("Neural Network Results:");
		EncogUtility.evaluate(network, trainingSet);
		
		Encog.getInstance().shutdown();
    }
    
}
