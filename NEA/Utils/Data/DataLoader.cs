using System;

namespace NEA.Utils.Data
{
   
    public class DataLoader
    {

        public Batch[] TrainSet;
        public Batch[] TestSet;
        public int BatchSize;

        /// <summary>
        /// Initialises a new DataLoader object with the specified parameters.
        /// Will always clean the dataset, regardless of whether or not it has already been cleaned. 
        /// </summary>
        /// <param name="dataset">The cleaned dataset to load data from.</param>
        /// <param name="batchSize">The size that each batch of data should take.</param>
        /// <param name="targetVariable">The column index(es) of the target variable(s).</param>
        /// <param name="oneHotTarget">Whether the target should be a one-hot vector. Default false</param>
        /// <param name="nClasses">The total number of classes for a one-hot vector. Default 0.</param>
        /// <param name="shuffle">Whether the dataset should be shuffled. Default true.</param>
        /// <param name="split">Whether the dataset should be split into train and test. Default true.</param>
        /// <param name="trainTestSplit">The proportion of the dataset that should be used as training data. Default 0.7.</param>
        public DataLoader(DataSet dataset, int batchSize, int[] targetVariable, bool oneHotTarget=false,int nClasses=0,bool shuffle=true, bool split=true, float trainTestSplit = 0.7f)
        {
            dataset.Clean();
            if (shuffle)
            {
                dataset.Shuffle();
            }
            if (split)
            {
                var (trainDataset, testDataset) = dataset.TrainTestSplit(trainTestSplit);
                this.TrainSet = batch(trainDataset, batchSize, targetVariable, oneHotTarget, nClasses);
                this.TestSet = batch(testDataset, batchSize, targetVariable, oneHotTarget, nClasses);
            }
            else
            {
                this.TrainSet = batch(dataset, batchSize, targetVariable, oneHotTarget, nClasses);
            }
            
        } 

        // Creates an array of batches from the dataset as defined by the parameters of the dataloader. 
        private Batch[] batch(DataSet dataset, int batchSize, int[] targetVariable, bool oneHotTarget, int nClasses)
        {
            int batchNo = (int)MathF.Ceiling(dataset.Count / batchSize); // number of batches

        }
    }
}