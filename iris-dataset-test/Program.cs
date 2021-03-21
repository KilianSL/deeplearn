using NEA.Utils.Data;
using NEA.NeuralNetwork;
using System;
using System.Linq;

namespace iris_dataset_test
{
    internal class Program
    {
        // Constants
        private const string DATASET_PATH = @"C:\Users\kilian\source\Git Repos\WCGS-2021-6C-seifk\iris-dataset-test\iris.csv";

        private const int BATCH_SIZE = 8;
        private const int N_CLASSES = 3;

        private static string arrayToString(float?[] arr)
        {
            string res = "[";
            foreach (var item in arr)
            {
                res = string.Concat(res, item.ToString() + ',');
            }
            res = string.Concat(res, ']');
            return res;
        }

        private static string arrayToString(float[] arr)
        {
            string res = "[";
            foreach (var item in arr)
            {
                res = string.Concat(res, item.ToString() + ',');
            }
            res = string.Concat(res, ']');
            return res;
        }

        private static void Main(string[] args)
        {
            // Load data from CSV
            var irisDataset = new DataSet(DATASET_PATH);
            Console.WriteLine("*************************************** Loaded Data ***************************************");
            Console.WriteLine("Iris Dataset Test \n Dataset Size: {0} \n Dataset Features: {1} \n", irisDataset.Count, irisDataset.Features);

            // Testing DataSet Methods
            Console.WriteLine("1st item: {0}", arrayToString(irisDataset[0]));
            Console.WriteLine("*************************************** Shuffling Dataset ***************************************");
            // Shuffle
            irisDataset.Shuffle();
            Console.WriteLine("New 1st item: {0} \n", arrayToString(irisDataset[0]));
            // Remove first item
            Console.WriteLine("*************************************** Removing item at index 0 ***************************************");
            irisDataset.RemoveElementAt(0);
            Console.WriteLine("New 1st item: {0} \n", arrayToString(irisDataset[0]));
            // Random Sample
            var sample1 = irisDataset.RandomSample(1);
            var sample2 = irisDataset.RandomSample(1);
            Console.WriteLine("Sample Selected: {0} \n {1} \n", arrayToString(sample1[0]), arrayToString(sample2[0]));
            // Setting one entry to be invalid to test the clean method
            irisDataset[3, 2] = null;
            // Clean Dataset
            int itemsRemoved = irisDataset.Clean();
            // Expects 1 item removed
            Console.WriteLine("Clean removed {0} items \n", itemsRemoved);

            // Create Dataloader
            var dataloader = new DataLoader(irisDataset, BATCH_SIZE, new int[] { 4 }, true, N_CLASSES);
            Console.WriteLine("Created DataLoader \n Training Dataset Length: {0} \n Test Dataset Length: {1} \n Batch Size: {2}", dataloader.TrainSet.Length * dataloader.BatchSize, dataloader.TestSet.Length * dataloader.BatchSize, dataloader.BatchSize);
            Console.WriteLine("1st Training Vector:");
            Console.WriteLine(dataloader.TrainSet[0].input.GetItem(0).ToString());
            Console.WriteLine("1st Output Vector:");
            Console.WriteLine(dataloader.TrainSet[0].output.GetItem(0).ToString());

            // Initialise neural network model
            var model = new Model();
            Console.WriteLine();
            Console.WriteLine("************************************ Starting Training ************************************");
            // Training Loop
            // Will only loop over dataset once, as backprop not yet implemented so there is no sense in using multiple epochs
            for (int i = 0; i < dataloader.TrainSet.Length; i++)
            {
                var batch = dataloader.TrainSet[i];
                var x = batch.input;
                var actualOutput = batch.output;

                Console.WriteLine("Training batch {0} of {1}", i + 1, dataloader.TrainSet.Length);
                Console.WriteLine("First input of batch:\n{0}", x[0].ToString());

                var prediction = model.Forward(x);
                float[] batchLosses = LossFunctions.CrossEntropyLoss(prediction, actualOutput);
                float averageLoss = batchLosses.Sum() / BATCH_SIZE;

                Console.WriteLine("Actual Output:\n{0}", actualOutput[0].ToString());
                Console.WriteLine("Predicted Output:\n{0}", prediction[0].ToString());
                Console.WriteLine("Batch Loss: {0}", averageLoss);
                Console.WriteLine();
            }
        }
    }
}