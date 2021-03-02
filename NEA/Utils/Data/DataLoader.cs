using System;
using System.Collections.Generic;

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
        public DataLoader(DataSet dataset, int batchSize, int[] targetVariable, bool oneHotTarget = false, int nClasses = 0, bool shuffle = true, bool split = true, float trainTestSplit = 0.7f)
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
            // Location function to turn a 1d array into a matrix-compatible 2d array
            float[,] arr1dTo2d(float[] arr)
            {
                float[,] res = new float[arr.Length, 1];
                for (int i = 0; i < arr.Length; i++)
                {
                    res[i, 0] = arr[i];
                }
                return res;
            }

            int batchNo = (int)MathF.Ceiling(dataset.Count / batchSize); // number of batches
            var outputList = new List<float[]>(); // each output vector
            var inputList = new List<float[]>(); // each input vector
            foreach (var item in dataset.ToArray()) // Generates 2 lists => one for the outputs and one for the inputs. These are "Linked" by index. 
            {
                if (oneHotTarget)
                {
                    var itemList = new List<float>(Array.ConvertAll(item, value => value.GetValueOrDefault()));
                    var output = new float[nClasses]; 
                    output[(int)item[targetVariable[0]].GetValueOrDefault()] = 1; // Creates one appropriate one-hot encoding
                    itemList.RemoveAt(targetVariable[0]);
                    outputList.Add(output);
                    inputList.Add(itemList.ToArray());
                }
                else
                {
                    var itemList = new List<float>(Array.ConvertAll(item, value => value.GetValueOrDefault()));
                    var output = new float[targetVariable.Length];
                    for (int outputidx = 0; outputidx < targetVariable.Length;  outputidx++)
                    {
                        output[outputidx] = item[targetVariable[outputidx]].GetValueOrDefault();
                        itemList.RemoveAt(targetVariable[outputidx]); // removes target variable from input
                    }
                    outputList.Add(output);
                    inputList.Add(itemList.ToArray());
                }
            }
            var batches = new List<Batch>();
            for (int batchidx = 0; batchidx < batchNo; batchidx++) // for each batch, creates a batch object and adds it to the list of batch objects for this dataset
            {
                var inputs = new Tensor.Tensor(batchSize, inputList[0].Length, 1);
                var outputs = new Tensor.Tensor(batchSize, outputList[0].Length, 1);
                for (int i = 0; i < batchSize; i++)
                {
                    inputs.SetItem(batchidx, new Tensor.Matrix(arr1dTo2d(inputList[batchidx * batchSize + i])));
                    outputs.SetItem(batchidx, new Tensor.Matrix(arr1dTo2d(outputList[batchidx * batchSize + i])));
                }
                batches.Add( new Batch() { input = inputs, output = outputs });
            }
            return batches.ToArray();
        }
    }
}