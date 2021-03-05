using System;
using NEA.Utils.Data;

namespace iris_dataset_test
{
    class Program
    {

        // Constants
        const string DATASET_PATH = @"C:\Users\kilian\source\Git Repos\WCGS-2021-6C-seifk\iris-dataset-test\iris.csv";
        const int BATCH_SIZE = 32;
        const int N_CLASSES = 3;

        static void Main(string[] args)
        {
            // Load data from CSV
            var irisDataset = new DataSet(DATASET_PATH);
            Console.WriteLine("Loaded Data");
            Console.WriteLine("Iris Dataset Test \n Dataset Size: {0} \n Dataset Features: {1}", irisDataset.Count, irisDataset.Features);

            // Clean Dataset
            int itemsRemoved = irisDataset.Clean();
            Console.WriteLine("Clean removed {0} items", itemsRemoved);

            // Create Dataloader
            var dataloader = new DataLoader(irisDataset, BATCH_SIZE, new int[]  { 4 }, true, N_CLASSES);
            Console.WriteLine("Created DataLoader \n Training Dataset Length: {0} \n Test Dataset Length: {1} \n Batch Size: {2}", dataloader.TrainSet.Length, dataloader.TestSet.Length, dataloader.BatchSize);
            Console.WriteLine("First Training Vector:");
            Console.WriteLine(dataloader.TrainSet[0].input.ToString());
        }
    }
}
